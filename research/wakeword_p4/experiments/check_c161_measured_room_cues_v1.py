"""Replay frozen PCM cues through training-only MIT responses, or verify saved inputs."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, lfilter

from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
import replay_c161_pcm_cues_v1 as matcher
import verify_c161_ambient_stream_v1 as live

REPORT = RESULTS / "4090-c161-measured-room-cues-v1.json"
MANIFEST = RESULTS / "4090-mit-rir-audit.json"
START = 14537


def room_inputs():
    frozen = read(RESULTS / "4090-c161-frozen-cue-prospective-plan-v1.json")["waves"]
    cues = {}
    for label, digest in frozen.items():
        path = RESULTS / f"hye-nyx-c161-synthetic-cue-{label}.wav"
        cues[label] = load_wav(dict(path=path, sha256=digest)).astype(float)
    assert set(cues) == {"v2", "v3"} and all(len(cue) == 21200 for cue in cues.values())
    dry_rms = np.sqrt(np.mean(cues["v2"] ** 2))
    sources = read(MANIFEST)["records"]
    assert len(sources) == len({s["path"] for s in sources}) == 270
    for i, source in enumerate(sources):
        assert source["split"] == "training"
        path = Path(".local/wakeword/mit-rirs") / source["path"]
        assert sha(path) == source["sha256"]
        response, rate = sf.read(path, dtype="float64")
        assert rate == 16000 and len(response) == source["samples"]
        response /= np.linalg.norm(response)
        peak = int(np.abs(response).argmax())
        wet = {label: fftconvolve(cue, response) for label, cue in cues.items()}
        # Both variants share the exact attenuation and additive noise waveform.
        level = min(.65, .80 * 32767 / max(np.max(np.abs(w)) for w in wet.values()))
        sample_count = START + len(wet["v2"]) + 8000
        rng = np.random.default_rng(130421 + i)
        noise = lfilter([1.], [1., -.85], rng.normal(size=sample_count))
        noise *= dry_rms * level / np.sqrt(np.mean(noise ** 2)) / 10
        for snr in (None, 20):
            for label, wave in wet.items():
                analog = np.r_[np.zeros(START), wave * level, np.zeros(8000)]
                end = START + peak + len(cues[label])
                wet_snr = None
                if snr is not None:
                    window = slice(START + peak, end)
                    wet_snr = 10 * np.log10(np.mean(analog[window] ** 2) / np.mean(noise[window] ** 2))
                    analog += noise
                pcm = np.rint(analog).clip(-32768, 32767).astype("<i2")
                meta = dict(response=source["path"], rir_sha256=source["sha256"], cue=label,
                            nominal_snr=snr, level=level, pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                            start=START, end=end, response_peak_delay=peak, source_samples=len(pcm),
                            clipped=int(((analog > 32767) | (analog < -32768)).sum()))
                yield meta, pcm, wet_snr


def verify_inputs():
    report = read(REPORT)
    assert report["rir_manifest_sha256"] == sha(MANIFEST)
    wet_levels = {"v2": [], "v3": []}
    for (meta, _, wet_snr), row in zip(room_inputs(), report["records"], strict=True):
        assert all(row[k] == value for k, value in meta.items()), meta
        if wet_snr is not None:
            wet_levels[meta["cue"]].append(wet_snr)
    for label, levels in wet_levels.items():
        print(label, "wet SNR dB (min, median, max)", np.percentile(levels, [0, 50, 100]))


def replay():
    assert sha(MODEL) == MODEL_SHA
    coarse = matcher.binding()
    matcher.LIB = matcher.LIB.with_name("pcm-suffix.so")
    confirmed = matcher.binding(True)
    native = binding()
    rows = []
    for row, pcm, _ in room_inputs():
        codes, clock, pair, _ = live.run_stream(native, pcm)
        base = times(decode(np.asarray(codes[0], np.uint8), np.asarray(clock)))
        pair_events = times(pair)
        proposals = matcher.detect(coarse, (pcm,))
        extra = matcher.detect(confirmed, (pcm,))
        row.update(original=base, pair=pair_events, proposals=proposals, confirmed=extra,
                   original_with_cue=matcher.joined(base, extra),
                   pair_with_cue=matcher.joined(pair_events, extra))
        rows.append(row)
    assert len(rows) == 1080
    save(REPORT, dict(rir_manifest_sha256=sha(MANIFEST), records=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pcm", action="store_true", help="rebuild inputs and verify every saved PCM hash")
    if parser.parse_args().verify_pcm:
        verify_inputs()
    else:
        replay()
