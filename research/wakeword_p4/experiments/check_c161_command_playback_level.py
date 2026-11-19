"""Measure source-side command level, microphone-only gain, and first-command retention."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from check_c161_v5_code_gain import CANDIDATE, PREFIX, RESULTS, binding, c161_binding, digest, load_reader, normalized
from check_c161_v5_code_gain import playback, read, routes, save, sha, transcripts, waves
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA

STEM = "4090-c161-command-playback-level"
REFERENCE = RESULTS/f"{PREFIX}-commands.json"
MAX_GAIN = 10**(12/20)
PEAK = 32767 * 10**(-1/20)


def linear(pcm):
    level = min(MAX_GAIN, PEAK/max(1, np.abs(pcm.astype(float)).max()))
    return np.rint(pcm * level).astype("<i2")


def limited(pcm):
    x = pcm.astype(float)
    allowed = np.minimum(MAX_GAIN, PEAK / np.maximum(1., np.abs(x)))
    # All 161 gains in the smoothing window have already seen the current sample.
    bound = minimum_filter1d(allowed, 321, mode="nearest")
    gain = uniform_filter1d(bound, 161, mode="nearest")
    wave = x * gain
    assert np.abs(wave).max() <= PEAK + 1e-6
    return np.rint(wave).astype("<i2")


def levels(pcm):
    x = pcm.astype(float) / 32768
    powers = np.mean(np.pad(x*x,(0,-len(x)%320)).reshape(-1,320),axis=1)
    active = powers >= powers.max() * .001
    return dict(peak_dbfs=float(20*np.log10(np.abs(x).max())),
                active_rms_dbfs=float(10*np.log10(powers[active].mean())), pcm_sha256=digest(pcm))


def originals():
    old, new = waves()
    report = read(REFERENCE)
    assert sha(MODEL) == MODEL_SHA == report["model_sha256"] and sha(CANDIDATE) == report["doubled_sha256"]
    with patch.object(playback, "TARGET", "v5"):
        paths = [playback.command_streams(new, noise_reference=old, process_command=op) for op in (None,linear,limited)]
        for (meta, raw), (same, peak), (other, compressed) in zip(*paths, strict=True):
            assert meta == same == other
            yield meta, {"baseline": raw, "linear": peak, "limited": compressed}


def recognition(reader, pcm, expected):
    result = transcripts(reader, pcm)
    got, want = normalized(result["text"]), normalized(expected)
    return dict(**result, first=got[:1]==want[:1], exact=got==want, complete_suffix=got[-len(want):]==want)


def primary(out):
    rows, controls = [], []
    reference = iter(read(REFERENCE)["records"])
    reader, identity = load_reader()
    native = c161_binding()
    with tempfile.TemporaryDirectory() as tmp:
        detector = binding(Path(tmp)/"v5.so", coded="v5")
        for meta, modes in originals():
            if meta["role"] == "v3":
                continue
            if meta["role"] != "v5":
                controls.append(dict(source=meta, routes={k:routes(detector,native,w) for k,w in modes.items()}))
                continue
            old = next(reference)
            assert old["source"]["voice"] == meta["voice"] and old["source"]["scene"] == meta["scene"]
            assert old["source"]["channel"] == meta["channel"] and old["doubled"]["pcm_sha256"] == digest(modes["baseline"])
            end = meta["end"]
            expected = old["expected"]
            row = dict(source=meta, expected=expected, modes={})
            for label, pcm in modes.items():
                # A resampled output immediately before the mapped boundary can
                # interpolate a changed command sample. Verify the preceding wake explicitly.
                np.testing.assert_array_equal(pcm[:end-2], modes["baseline"][:end-2])
                r = routes(detector,native,pcm)
                assert r["merged"] == old["doubled"]["merged"] and meta["start"] <= r["merged"][0] < end
                full_start = max(0, ((r["merged"][0]+255)//256*256)//160*160 - 32000)
                r.update(command=recognition(reader,pcm[end:],expected), raw_capture_start=full_start,
                         capture=recognition(reader,pcm[full_start:],expected),
                         command_level=levels(pcm[end:]))
                row["modes"][label] = r
            # Same gain processing after the acoustic noise enters the microphone.
            row["receiver_limited"] = recognition(reader,limited(modes["baseline"][end:]),expected)
            assert row["modes"]["baseline"]["command"]["text"] == old["doubled"]["command"]["text"]
            rows.append(row)
    assert next(reference,None) is None and len(rows)==36 and len(controls)==72
    save(out,dict(source_report_sha256=sha(REFERENCE),cue_sha256=sha(CANDIDATE),model_sha256=MODEL_SHA,
                  asr_files=identity,records=rows,controls=controls))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=RESULTS/f"{STEM}.json")
    primary(p.parse_args().out)
