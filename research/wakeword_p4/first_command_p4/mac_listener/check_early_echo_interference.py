"""Stress the 180 ms extra check with previously consumed everyday and background recordings."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import time_calibrated_echo_names as timing

ch, spoken = timing.ch, timing.spoken
base = timing.timing.base
OUT = ch.RESULTS / (timing.STEM + "-interference.json")


def feature_prefix(native, pcm, end):
    # Rebuild frontend history once. Long files need only the raw frames through their last checked window.
    samples = base.PRIME + end
    frames = np.zeros(((samples + 159)//160, 160), np.int16)
    frames.ravel()[base.PRIME:samples] = pcm[:end]
    raw = np.empty((len(frames)-2, 40), np.uint16)
    quiet = np.zeros(len(frames), np.uint8)
    assert native(frames.ctypes.data, quiet.ctypes.data, len(frames), 0, raw.ctypes.data, raw.size) == 0
    return raw


def run():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    net = spoken.natural.Detector()
    history = ch.RESULTS / (timing.timing.study.training.STEM + "-replay.json")
    reference = ch.read(history)
    times = ch.read(timing.names.OUT)
    assert net.identity == times["models"] and net.cutoff == reference["calibrated_cutoff"]
    reader, identity = ch.load_reader()
    assert identity == reference["asr"] == times["asr"]
    sources = {history.name:ch.sha(history), timing.names.OUT.name:ch.sha(timing.names.OUT)}
    groups, summary = {}, {}
    for group in ("daily", "bot", "music", "background"):
        path = ch.RESULTS / (spoken.natural.replay.fit.STEM + "-" + group + ".json")
        sources[path.name] = ch.sha(path)
        archived = ch.read(path)["records"]
        records, total = [], Counter()
        for (meta, pcm, independent, _, _), old, prior in zip(base.sources(group), archived, reference["records"][group], strict=True):
            assert ch.digest(pcm) == old["pcm_sha256"] == prior["pcm_sha256"]
            assert len(pcm) == old["samples"] and independent == old["events"]["c161"] and old["index"] == prior["index"]
            proposals = [t for t in old["proposals"]["early"] if 0 < t <= len(pcm)]
            ends = sorted({t+d for t in proposals for d in (timing.EARLY, timing.STANDARD) if t+d <= len(pcm)})
            raw = feature_prefix(net.native, pcm, ends[-1]) if ends else None
            checks = spoken.natural.replay.verifier(raw, ends, net.student, net.matcher, net.gap)
            windows = {}
            for t, w in checks.items():
                cost = times["char_us"]
                passed = False
                if w["accepted"]:
                    patch = np.rint(spoken.natural.replay.prior.fit.room.window(raw, t).clip(0,666)*(255/666)).astype(np.uint8)
                    w["five"] = net.models["trained"].invoke(patch).tolist()
                    cost += times["five_us"]
                    passed = w["five"][0] >= net.cutoff
                windows[t] = dict(service_us=cost, id=t, passes=passed)
            # Check all standard windows independently of the early decision.
            admitted = []
            for t in proposals:
                end = t + timing.STANDARD
                if end <= len(pcm):
                    w = checks[end]
                    assert w["output_sha256"] == old["checks"][str(end)]["output_sha256"]
                    assert w["accepted"] == (str(end) in prior["checks"])
                    if w["accepted"]:
                        assert w["five"] == prior["checks"][str(end)]["trained"]
                    if windows[end]["passes"]:
                        admitted.append(end)
            assert base.scoring.merged(independent, admitted) == prior["events"]["calibrated"]
            row = dict(start=0, end=len(pcm), command_at=0, stages=dict(raw=dict(c161=independent),
                       filtered=dict(c161=[], proposals=proposals)))
            entry = dict(index=prior["index"], samples=len(pcm), pcm_sha256=old["pcm_sha256"], source=meta,
                         independent=independent, checks=checks, proposals=proposals, routes={})
            texts = {}
            for name, delays in (("standard",(timing.STANDARD,)), ("early",(timing.EARLY, timing.STANDARD))):
                route = timing.schedule(row, windows, delays, 0)
                events = route["events"]
                spans = meta["source"]["spans"] if group in ("daily","bot") else []
                entry["routes"][name] = dict(events=events, jobs=route["jobs"], spans=[
                    dict(text=s["text"], role=s["address"], hits=[e for e in events if s["start"] <= e < s["end"]+6400])
                    for s in spans])
                if group in ("music","background"):
                    for t in events:
                        if t not in texts:
                            clip = pcm[max(0,t-80000):min(len(pcm),t+48000)]
                            texts[t] = dict(pcm_sha256=ch.digest(clip), reading=timing.names.transcripts(reader,clip))
                total[name + "_events"] += len(events)
                total[name + "_windows"] += len(route["jobs"])
            entry["readings"] = texts
            records.append(entry)
            total.update(trials=1, samples=len(pcm))
            if group == "background" and len(records)%10 == 0:
                print(group,len(records),flush=True)
        groups[group], summary[group] = records, total
        print(group, dict(total),flush=True)
    ch.save(OUT, dict(records=groups, summary=summary, sources=sources, models=net.identity, asr=identity,
            script_sha256=ch.sha(Path(__file__)), schedule_sha256=ch.sha(Path(timing.__file__)),
            policy="Apply the same two frozen schedules to full historical microphone inputs with archived "
            "early-proposer and independent C161 clocks. Verify standard decisions and five-class outputs. "
            "Use previously reported QEMU P95 service times. This tests everyday false openings if enabled "
            "on the microphone; no echo filter or panel-only calibration on these inputs. Development reuse. "
            "Hours refer to the archived background only. ASR excerpts describe the surrounding audio."))


if __name__ == "__main__":
    run()
