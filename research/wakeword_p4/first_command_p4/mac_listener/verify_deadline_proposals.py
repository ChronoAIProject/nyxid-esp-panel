"""Apply the existing five-class verifier to the deadline models' accepted spelling windows."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import replay_proposal_deadlines as replay
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_spelling_p4_commands import outcome

base, fit, prior = replay.base, replay.fit, replay.prior


def main(group):
    path = RESULTS/f"{fit.STEM}-{group}.json"
    spec = read(RESULTS/"4090-c161-positive-readout-exports.json")["models"]["trigger"]
    file = RESULTS/spec["file"]
    assert sha(file) == spec["sha256"]
    five = base.poll.Model(file)
    native = prior.fit.native_frontend()
    scene = group in ("acoustics","matched")
    reader, identity = base.scoring.word.load_reader() if scene or group=="dialogue" else (None,None)
    sums, records = defaultdict(Counter),[]
    old = read(path)
    archive = read(RESULTS/(replay.gap.STEM+"-comparison.json"))["records"][group] if scene else None
    dialogue_plan = read(replay.probe.dialogue.corpus.PLAN)["records"] if group=="dialogue" else None
    for (idx,pcm,c161,meta,spans), r in zip(replay.sources(group),old["records"],strict=True):
        assert r["index"] == idx and r["events"]["c161"] == c161 and r["pcm_sha256"] == base.poll.digest(pcm)
        kept = [int(t) for t,v in r["checks"].items() if v["accepted"]]
        raw = base.native_inputs(native,pcm)[0] if kept else None
        outputs = {}
        for t in kept:
            patch = np.rint(prior.fit.room.window(raw,t).clip(0,666)*(255./666)).astype(np.uint8)
            outputs[t] = five.invoke(patch).tolist()
        events = {}
        for mode in fit.MODES:
            accepted = [p+base.scoring.word.WAIT for p in r["proposals"][mode] if p+base.scoring.word.WAIT in outputs]
            assert base.scoring.merged(c161,accepted)==r["events"][mode]
            events[mode] = base.scoring.merged(c161,[t for t in accepted if outputs[t][0]>=0])
        row = dict(index=idx,pcm_sha256=r["pcm_sha256"],five=outputs,events=events)
        if scene:
            cache = {int(k):v for store in (archive[idx]["captures"],r["new_transcripts"]) for k,v in store.items()}
            previous = set(cache)
            row["scene"] = {k:outcome(ts,meta,pcm,reader,cache) for k,ts in events.items()}
            row["new_transcripts"] = {k:v for k,v in cache.items() if k not in previous}
        if group == "dialogue":
            src = dialogue_plan[idx]
            assert src["pcm_sha256"] == r["pcm_sha256"]
            row["captures"] = replay.probe.dialogue.piper.annotate(src,pcm,events,[],[],reader)
            for span in row["captures"]["rows"]:
                if span["address"] in ("Hey Nyx","Hye Nyx","Hey Nicks"):
                    for k,found in span["routes"].items():
                        sums[k]["target.complete"] += bool(found.get("exact_command"))
        for k,ts in events.items():
            counter = sums[k]
            counter["samples"] += len(pcm)
            counter["events"] += len(ts)
            if scene:
                kind = meta["role"] if group=="matched" else meta["acoustic"]
                result = row["scene"][k]
                counter[kind+".trials"] += 1
                counter[kind+".timely"] += result["timely"]
                counter[kind+".complete"] += result.get("complete_suffix",False)
            elif spans is not None:
                replay.tally(ts,spans,group,counter)
            elif group=="joe":
                counter[meta["label"]+".hits"] += bool(ts)
                counter[meta["label"]+".trials"] += 1
        records.append(row)
    save(RESULTS/f"{fit.STEM}-five-{group}.json",dict(records=records,summary=sums,source_sha256=sha(path),
         model_sha256=sha(file),asr=identity,script_sha256=sha(Path(__file__)),
         policy="Require the frozen five-class model's signed Nyx score >=0 in the same window that passed "
                "spelling and the gap rule. C161 remains independently eligible. Audio-availability clocks."))
    print(group,{k:v["events"] for k,v in sums.items()},flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("groups",nargs="+",choices=[*replay.gap.study.GROUPS,"joe","dialogue"])
    for group in p.parse_args().groups:
        main(group)
