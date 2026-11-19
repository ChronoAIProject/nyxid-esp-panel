"""Replay the frozen 180 ms gate on ordinary synthetic voices, continuous dialogue and paired Willow inputs."""
from collections import Counter, defaultdict
from pathlib import Path

import torch

import score_early_echo_gate as gate
import replay_early_proposal_readout as study
import connected_nick_willow as connected

ch, timing = gate.ch, gate.timing
STEM = "4090-c161-early-gate-voices"
OUT = ch.RESULTS / (STEM + ".json")
GROUPS = ("acoustics", "matched", "joe", "dialogue", "fresh", "context", "paired", "connected")
HISTORY = ch.RESULTS / (study.training.STEM + "-replay.json")
PAIRED = ch.RESULTS / (study.matched.STEM + ".json")
CONNECTED = ch.RESULTS / (connected.STEM + ".json")


def inputs(group, prior):
    if group == "connected":
        history = ch.read(CONNECTED)
        assert ch.sha(connected.PLAN) == history["plan_sha256"]
        for src, r in zip(ch.read(connected.PLAN)["records"],history["records"],strict=True):
            pcm = study.replay.probe.dialogue.piper.load_wav(src)
            assert ch.digest(pcm) == r["pcm_sha256"] == src["pcm_sha256"] and src["index"] == r["index"]
            meta = dict(role=src["target_id"],voice=src["voice"],cue_at=src["start"],command_at=src["end"],expected=src["carrier"])
            yield r["index"], pcm, meta, None, r["routes"]["c161"]["events"], r["proposals"], r["routes"]["parent"]["events"]
    else:
        for item, old in zip(study.inputs(group),prior[group],strict=True):
            i,pcm,direct,meta,spans,ref,_,_ = item
            assert i == old["index"] and ch.digest(pcm)==old["pcm_sha256"]
            proposals = [t for t in ref["proposals"]["early"] if 0<t<=len(pcm)]
            yield i,pcm,meta,spans,direct,proposals,old["events"]["calibrated"]


def score(detector, pcm, direct, proposals, standard, spec, costs):
    early = [t+timing.EARLY for t in proposals if t+timing.EARLY <= len(pcm)]
    trace = detector.score(pcm,early)
    assert trace["proposals"] == proposals and trace["c161"] == direct and trace["events"] == standard
    windows, features = {}, {}
    for t,w in trace["checks"].items():
        cost = costs["char_us"]
        if w["accepted"]:
            features[t] = gate.fit.features(w,w["five"])
            cost += costs["five_us"]
        windows[t] = dict(id=t,service_us=cost,passes=w["pass"])
    row = dict(start=0,end=len(pcm),command_at=0,stages=dict(raw=dict(c161=direct),filtered=dict(c161=[],proposals=proposals)))
    runs = gate.routes(row, windows, features, spec, 0)
    events = {k:v["events"] for k,v in runs.items()}
    # Final events are delivered at the next microphone read boundary, including independent C161.
    events["c161"] = [timing.block(t) for t in direct]
    events = {k:[timing.block(t) for t in ts] for k,ts in events.items()}
    return events, trace["checks"], {k:len(v["jobs"]) for k,v in runs.items()}


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    detector = timing.spoken.natural.Detector()
    spec = ch.read(gate.fit.COEFF)
    assert detector.identity == spec["models"] and ch.sha(gate.fit.FIT) == spec["fit_sha256"]
    costs = ch.read(timing.names.OUT)
    prior = ch.read(HISTORY)["records"]
    stats = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    records = defaultdict(list)
    for group in GROUPS:
        for i,pcm,meta,spans,direct,proposals,standard in inputs(group,prior):
            events,checks,jobs = score(detector,pcm,direct,proposals,standard,spec,costs)
            if group == "paired":
                meta = dict(meta, voice=meta["cue"])
            role = meta.get("role", meta.get("label", meta.get("acoustic",group)))
            for k,times in events.items():
                total = stats[group][role][k]
                total.update(trials=1,hits=bool(times),events=len(times),jobs=jobs.get(k,0))
                if "command_at" in meta:
                    total["timely"] += bool(times and meta["cue_at"] <= times[0] < meta["command_at"])
                if spans is not None:
                    study.replay.tally(times,spans,group,total)
            records[group].append(dict(index=i,source=meta,spans=spans,pcm_sha256=ch.digest(pcm),samples=len(pcm),
                                        c161=direct,proposals=proposals,checks=checks,routes=events,windows=jobs))
        print(group,len(records[group]),flush=True)
    paths = [HISTORY,gate.fit.COEFF,timing.names.OUT,PAIRED,CONNECTED,study.matched.PLAN,connected.PLAN,
             Path(__file__),Path(gate.__file__)]
    ch.save(OUT,dict(records=records,summary=stats,models=detector.identity,device=torch.cuda.get_device_name(),
        sources={p.name:ch.sha(p) for p in paths},
        policy="Rebuild the raw microphone C161 and proposer traces, replay trained verifiers at 180 and 360 ms, "
        "check previous 360 ms events before charging time. Each complete window waits for its 256-sample input "
        "block. Use prior serial component P95 times and deliver on the next read block. Independent C161 has "
        "its own clock. All eight cohorts already consumed. Role-specific timeliness uses the first OPEN."))


if __name__ == "__main__":
    main()
