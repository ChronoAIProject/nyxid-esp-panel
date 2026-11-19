"""Replace each following command by silence while leaving the cue and its frame phase unchanged."""
from collections import Counter, defaultdict
from pathlib import Path

import torch

import score_reusable_nyx as replay

ch, voices, study = replay.ch, replay.voices, replay.study
OUT = ch.RESULTS / (study.STEM + "-suffix.json")


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    parent = ch.read(replay.OUT)
    assert parent["plan_sha256"] == ch.sha(study.PLAN)
    spec = ch.read(voices.gate.fit.COEFF)
    fit = ch.read(replay.veto.fit.FIT)
    costs = ch.read(voices.timing.names.OUT)
    net = voices.timing.spoken.natural.Detector()
    assert net.identity == parent["models"] == fit["models"] == spec["models"]
    records = []
    summary = defaultdict(lambda: defaultdict(Counter))
    for (s, pcm), prev in zip(study.inputs(), parent["records"], strict=True):
        assert ch.digest(pcm) == prev["pcm_sha256"]
        if s["pair"] is None:
            continue
        pcm[s["cue_end"]:] = 0
        assert ch.digest(pcm[:s["cue_at"]]) == s["prefix_sha256"]
        raw = net.score(pcm)
        modes, checks, _ = voices.score(net, pcm, raw["c161"], raw["proposals"], raw["events"], spec, costs)
        stage = dict(raw=dict(c161=raw["c161"]), filtered=dict(c161=[], proposals=raw["proposals"]))
        state = dict(start=s["cue_at"], end=len(pcm), command_at=s["cue_end"], stages=stage)
        events = replay.veto.evaluate(state, checks, spec, {"paired": fit["heads"]["paired"]}, costs)
        assert events["prior"] == modes["linear"]
        events["c161"] = modes["c161"]
        result = dict(index=prev["index"], cue=s["cue"], pair=s["pair"], pcm_sha256=ch.digest(pcm), events=events)
        for route, silence in events.items():
            with_speech = prev["routes"][route]["events"]
            end = s["cue_end"]
            assert [t for t in silence if t <= end] == [t for t in with_speech if t <= end]
            summary[s["role"]+"."+s["style"]][route].update(
                trials=1, both=bool(with_speech and silence), speech_only=bool(with_speech and not silence),
                silence_only=bool(silence and not with_speech), neither=not (with_speech or silence),
                same_first=bool(with_speech and silence and silence[0] == with_speech[0]))
        records.append(result)
    assert len(records) == 288
    ch.save(OUT, dict(records=records, summary=summary, parent_sha256=ch.sha(replay.OUT),
                      script_sha256=ch.sha(Path(__file__)), plan_sha256=ch.sha(study.PLAN),
                      policy="All 288 exposed command slots. Replace every sample of the following command "
                      "with digital zero, preserving prefix, cue placement, sample count, and recognition clocks. "
                      "Match all delivered events up through the boundary, then count slots opening under "
                      "both suffixes, speech alone, silence alone, or neither. No ASR on the silence condition."))


if __name__ == "__main__":
    main()
