"""Test whether proposals rejected during Bot speech suppress the next genuine request."""
from pathlib import Path

import numpy as np

import check_spoken_playback_echo as spoken
import replay_spoken_echo_frontends as front
from run_c161_frontend_v5 import parse_uart
from score_c161_lexical_echo import subtract_playback
import check_c161_speex_echo as speex

ch = spoken.ch
OUT = ch.RESULTS / "4090-c161-spoken-proposal-cooldown.json"


def check(detector, pcm, previous):
    base = spoken.natural.base
    word = base.scoring.word
    _, bands, clock = base.native_inputs(detector.native, pcm)
    trace = base.infer(detector.front, bands)
    sums = np.convolve(trace.astype(np.int32), np.ones(5, dtype=np.int32), mode="valid")
    armed = True
    crossings = []
    for i, value in enumerate(sums, 4):
        if 2 * value < 1175:
            armed = True
        if armed and value >= 1175:
            crossings.append(int(clock[i]))
            armed = False
    proposals = [t for t in crossings if 0 < t <= len(pcm) - word.WAIT]
    ends = [t + word.WAIT for t in proposals]
    original = detector.score(pcm, ends)
    assert original["events"] == previous
    passed = [t for t in ends if original["checks"][t]["pass"]]
    events = base.scoring.merged(original["c161"], passed)
    return dict(original=previous, original_proposals=original["proposals"], crossings=proposals,
                checked=len(ends), original_checked=len(original["live_ends"]), accepted=passed, events=events,
                scores_sha256=ch.digest(trace), clocks_sha256=ch.digest(clock))


def main():
    import torch
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    detector = spoken.natural.Detector()
    parent = ch.read(spoken.OUT)
    prior = ch.read(front.OUT)
    assert parent["models"] == prior["models"] == detector.identity
    cache = {}
    def evaluate(pcm, expected):
        key = ch.digest(pcm)
        if key not in cache:
            cache[key] = check(detector, pcm, expected)
        assert cache[key]["original"] == expected
        return key
    known = []
    for group in ("daily", "bot"):
        history = (r for r in parent["known"] if r["group"] == group)
        for (_, pcm, _, _, _), row in zip(spoken.natural.base.sources(group), history, strict=True):
            assert ch.digest(pcm) == row["mic_sha256"]
            known.append(dict(group=group, index=row["index"], signal=evaluate(pcm,row["natural"]),
                              spans=row["source"]["spans"]))
    by_pair = {(r["mic_sha256"], r["ref_sha256"]): r for r in prior["records"] if r["group"] == "additional_pairs"}
    build = front.build_id(front.firmware())
    api = speex.binding()
    records = []
    for meta, mic, ref in spoken.extra_cases():
        if meta.get("role") != "Hey Nyx" and meta["mode"] != "self_only":
            continue
        row = by_pair[ch.digest(mic), ch.digest(ref)]
        audio = parse_uart(front.log_path(row, build), ("echo",))["echo"][4]
        residual, _ = subtract_playback(mic, ref)
        audio.update(mic=mic, fir=residual, speex=speex.cancel(api, mic, ref))
        stages = {}
        for stage, pcm in audio.items():
            digest = ch.digest(pcm)
            assert digest == row["stages"][stage]["pcm_sha256"]
            stages[stage] = evaluate(pcm, prior["signals"][digest]["events"])
        records.append(dict(source=meta, start=row["start"], end=row["end"], command_at=row["command_at"], stages=stages))
    ch.save(OUT, dict(known=known, records=records, signals=cache, models=detector.identity,
                      sources={p.name:ch.sha(p) for p in (front.OUT, spoken.OUT, Path(__file__))},
                      policy="Keep the 5-frame sum, 1175 crossing, low-score rearm and all trained models. "
                      "Run the existing verifier 360 ms after every new crossing. Remove the 67-step wait "
                      "on proposals only. Preserve the existing final notification cooldown and independent C161. "
                      "Reproduce old events for every waveform. No physical timing or additional acceptance exposure."))


if __name__ == "__main__":
    main()
