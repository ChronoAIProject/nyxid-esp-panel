"""Score sign agreement at real reference-pair events and across Opus transports."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

import replay as listener
from check_c161_v5_code_gain import RESULTS, waves
from check_c161_v5_playback_opus import codec_api, roundtrip
from replay_c161_pcm_cues_v1 import detect


@lru_cache(maxsize=1)
def pattern():
    original, _ = waves()
    return original[14800:15312:4] > 0


def distance(ref, at):
    indices = at - np.arange(832, 929)[:, None] + np.arange(128)[None, :]*4
    bits = (ref[indices.clip(0, len(ref)-1)] > 0) & (indices >= 0) & (indices < len(ref))
    counts = (bits != pattern()).sum(1)
    return int(np.minimum(counts, 128-counts).min())


def qualify(ref, t):
    return distance(ref, t) <= 12


def verify_backgrounds(pair):
    rows = []
    sources = {}
    for group in ("primed", "jenny", "melo"):
        report = RESULTS/f"4090-c161-coded-cue-aligned-v4-background-{group}.json"
        archived = listener.read(report)
        assert archived["header_sha256"] == listener.sha(RESULTS.parent/"portable/c161_coded_cue_aligned_v4.h")
        assert archived["coefficients_sha256"] == listener.sha(RESULTS.parent/"portable/c161_coded_cue_band_coefficients_v4.h")
        sources[group] = dict(report_sha256=listener.sha(report), samples=archived["samples"])
        paths = {r["source"]["pcm_sha256"]: r["source"] for r in
                 listener.read(RESULTS/f"4090-c161-history-gain-{group}-v1.json")["records"] if r["phase"] == "background"}
        for r in archived["records"]:
            if not r["events"]:
                continue
            src = paths[r["pcm_sha256"]]
            path = Path(src.get("wav_path") or src["path"])
            with sf.SoundFile(path) as f:
                assert f.channels == 1 and f.samplerate == 16000 and f.subtype == "PCM_16" and len(f) == src["samples"]
                for event in r["events"]:
                    t = event["t"]
                    start = max(0, t-48000)
                    f.seek(start)
                    ref = f.read(t-start+1, dtype="int16")
                    assert t-start in [e["t"] for e in detect(pair, [ref])]
                    rows.append(dict(group=group, source_id=r["source_id"], source_pcm_sha256=r["pcm_sha256"],
                        excerpt_sha256=listener.digest(ref), at=t, start=start, sign_errors=distance(ref, t-start)))
    return dict(reused=sources, records=rows)


def transports(pair, full):
    rows = []
    decoder = codec_api()
    old_report = listener.read(RESULTS/"4090-c161-v5-playback-opus.json")
    for label, cue in zip(("v5", "doubled"), waves(), strict=True):
        modes = {"pcm": cue}
        for config in old_report["records"]:
            if config["mode"] == "raw":
                continue
            pcm = roundtrip(decoder, cue, config["application"], config["bitrate"])[0]
            if label == "v5":
                assert listener.digest(pcm) == config["cue_pcm_sha256"]
            modes[config["mode"]] = pcm
        for mode, wave in modes.items():
            for gain in (1., -.25, .02):
                ref = np.r_[np.zeros(14537, np.int16), np.rint(wave.astype(float)*gain).astype("<i2"), np.zeros(4000,np.int16)]
                prefix = [e["t"] for e in detect(pair, [ref])]
                rows.append(dict(wav=label, mode=mode, gain=gain, reference_sha256=listener.digest(ref),
                    complete=[e["t"] for e in detect(full, [ref])], prefix=prefix, errors=[distance(ref, t) for t in prefix]))
    return rows
