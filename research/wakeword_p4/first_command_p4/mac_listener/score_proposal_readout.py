"""Compare INT8 readouts at every frozen music-proposal clock on consumed development audio."""
import argparse
import ctypes
from pathlib import Path

import numpy as np

import train_proposal_readout as fit
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = fit.replay
GROUPS = ("acoustics","matched","fresh","context","daily","bot","music","background")


def main(group):
    parent = RESULTS/f"{fit.music.STEM}-{group}.json"
    old = read(parent)
    exported = RESULTS/(fit.STEM+"-exports.json")
    spec = read(exported)
    assert spec["fit_sha256"] == sha(fit.FIT)
    assert old["word_sha256"] == read(fit.FIT)["front_sha256"] and old["c161_sha256"] == base.scoring.MODEL_SHA
    parents, hashes = base.scoring.verifiers()
    assert old["verifier_sha256"] == hashes["teacher"]
    models = {"parent":parents["teacher"]}
    for mode in fit.MODES:
        item = spec["models"][mode]
        path = RESULTS/item["file"]
        assert sha(path) == item["sha256"]
        models[mode] = base.poll.Model(path)
    lib = base.BUILD/"frontend.so"
    assert sha(lib) == old["native_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    reader, identity = base.scoring.word.load_reader() if group in ("acoustics","matched","fresh","background") else (None,None)
    rows = []
    for (meta, pcm, c161, _, _), prev in zip(base.sources(group),old["records"],strict=True):
        digest = base.poll.digest(pcm)
        assert digest == prev["pcm_sha256"] and c161 == prev["c161"] and len(pcm) == prev["samples"]
        ends = [(t,t+base.scoring.word.WAIT) for t in prev["proposals"] if 0<t<=len(pcm)-base.scoring.word.WAIT]
        checks = []
        if ends:
            raw, bands, clock = base.native_inputs(native,pcm)
            assert base.poll.digest(clock) == prev["clock_sha256"]
            del bands
            for (t,end),c in zip(ends,prev["checks"],strict=True):
                assert (t,end) == (c["proposal"],c["end"])
                patch = np.rint(fit.room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
                codes = {k:v.invoke(patch).tolist() for k,v in models.items()}
                assert codes["parent"] == c["codes"]
                checks.append(dict(proposal=t,end=end,codes=codes))
            del raw
        assert len(checks) == len(prev["checks"])
        routes = {k:base.scoring.merged(c161,[c["end"] for c in checks if c["codes"][k][0]>=0]) for k in models}
        assert routes["parent"] == prev["events"]
        row = dict(index=len(rows),samples=len(pcm),pcm_sha256=digest,c161=c161,proposals=prev["proposals"],checks=checks,routes=routes)
        if group in ("acoustics","matched"):
            captures = {}
            start, end = meta["cue_at"], meta["command_at"]
            for k, events in routes.items():
                if events and start<=events[0]<end and meta.get("role","ordinary") in ("ordinary","coded","willow"):
                    begin = max(0,((events[0]+255)//256*256)//160*160-32000)
                    if begin not in captures:
                        captures[begin] = prev["capture"] if begin == prev.get("begin") else base.scoring.recognition(reader,pcm[begin:],meta["expected"])
            row.update(source=meta, captures=captures)
        elif group == "fresh":
            row.update(base.poll.annotate(meta["source"],pcm,routes,[],[],reader))
            row["source"] = {k:v for k,v in meta["source"].items() if k != "timeline"}
        elif group == "background":
            saved = {v["sample"]:v for v in prev["false_notifications"]}
            for t in sorted(set().union(*routes.values())-saved.keys()):
                lo,hi = max(0,t-80000),min(len(pcm),t+48000)
                saved[t] = dict(sample=t,begin=lo,end=hi,asr=base.study.diagnostic.scene.transcripts(reader,pcm[lo:hi]))
            row.update(key=prev["key"], context=list(saved.values()))
        else:
            row["source"] = meta["source"]
        rows.append(row)
        if group == "background" and len(rows)%10 == 0:
            print(group,len(rows),flush=True)
    save(RESULTS/f"{fit.STEM}-{group}.json", dict(records=rows, previous_sha256=sha(parent), exports_sha256=sha(exported),
        parent_sha256=old["verifier_sha256"], script_sha256=sha(Path(__file__)), asr=identity,
        policy="Replay every complete archived INT8 proposal on the same continuous input. Check parent INT8 outputs "
               "and all merged first-notification clocks. Frozen C161 union and original cooldown; re-read source "
               "PCM for both new INT8 heads, re-transcribe changed capture clocks where scored."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="+", choices=GROUPS)
    for group in p.parse_args().groups:
        main(group)
