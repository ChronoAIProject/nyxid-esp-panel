"""Replay each newly aligned INT8 model on the full existing target, speech and background streams."""
import argparse
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import post120_command_uplinks as commands
import replay_unmarked_dwell as dwell
import train_connected_alignment as training

ch, score = dwell.ch, dwell.score
OUT = ch.RESULTS / (training.STEM+"-replay.json")
BG = ch.RESULTS / (training.STEM+"-background.json")
COMMANDS = ch.RESULTS / (training.STEM+"-commands.json")


def background(previous):
    for r, (meta, pcm, c161, clock_sha, _) in zip(previous["records"],score.base.sources("background"),strict=True):
        assert meta["key"] == r["key"] and clock_sha == r["clock_sha256"] and not c161
        yield dict(r,group="background"),pcm,None,None


def replay(long):
    parent_path = dwell.background.OUT if long else dwell.replay.OUT
    previous = ch.read(parent_path)
    short = ch.read(dwell.replay.OUT) if long else previous
    streams = background(previous) if long else islice(dwell.materials(short,{}),len(short["records"]))
    native = score.fit.prior.letters.native_frontend()
    five, five_ids = score.verifier.models()
    assert five_ids["trained"] == short["five_sha256"]
    models, hashes = {}, {}
    for arm in training.ARMS:
        stem = training.STEM+"-"+arm
        path = ch.RESULTS/(stem+"-v1.tflite")
        hashes[arm] = ch.sha(path)
        assert hashes[arm] == ch.read(ch.RESULTS/(stem+"-export-v1.json"))["sha256"]
        model = score.REF_KERNEL(model_path=str(path))
        model.allocate_tensors()
        models[arm] = model
    arrays, rows = {}, []
    out = BG if long else OUT
    for old,pcm,raw,echo in streams:
        key = old["key"]
        assert len(pcm) == old["samples"] and ch.digest(pcm) == old["pcm_sha256"]
        bands, pooled, clock = score.base.native_inputs(native,pcm)
        if long:
            assert ch.digest(clock) == old["clock_sha256"]
        arrays[key+"_clock"] = clock
        proposals, checks = {}, {}
        for arm, model in models.items():
            codes = score.base.infer(model,pooled)
            events = [t for t in score.times(score.decode(codes,clock)) if t>0]
            assert events == dwell.fit.detect(codes,clock,5,1175)
            arrays[key+"_"+arm] = codes
            proposals[arm] = events
            for t in events:
                if t not in checks:
                    window = score.verifier.training.room.window(bands,t//160*160)
                    x = np.rint(window.clip(0,666)*(255/666)).astype(np.uint8)
                    checks[t] = five["trained"].invoke(x).tolist()
            proposals[arm+"_confirmed"] = [t for t in events if checks[t][0] >= short["cutoff"]]
        r = {k:old[k] for k in ("key","group","cohort","track","source","text","voice","speed","samples","pcm_sha256") if k in old}
        r.update(proposals=proposals,name_scores=checks)
        if raw is not None:
            s = r["source"]
            opened = {}
            for name,hits in proposals.items():
                combined = dict(raw,c161=sorted(set(raw["c161"]+hits)))
                opened[name] = score.mixed.panel.middle.pair.admit(combined,s["transmit_event"],echo,np.zeros_like(pcm))["opened"]
            r["opened"] = opened
        elif long:
            r["events"] = proposals
        else:
            r["events"] = {name:score.base.scoring.merged(old["events"]["baseline"],hits) for name,hits in proposals.items()}
        rows.append(r)
        if long and len(rows)%10 == 0:
            print("background",len(rows),flush=True)
    assert len(rows) == (66 if long else 308) and len({r["key"] for r in rows}) == len(rows)
    with TemporaryDirectory(dir=out.parent) as tmp:
        work = Path(tmp)
        archive = out.with_suffix(".npz")
        np.savez_compressed(work/archive.name, **arrays)
        with np.load(work/archive.name,allow_pickle=False) as z:
            for k,v in arrays.items():
                np.testing.assert_array_equal(z[k],v)
        ch.save(work/out.name,dict(models=hashes,records=rows,parent_sha256=ch.sha(parent_path),
                c161=short["c161"], five_sha256=short["five_sha256"], cutoff=short["cutoff"],
                script_sha256=ch.sha(Path(__file__)), archive_sha256=ch.sha(work/archive.name)))
        (work/archive.name).replace(archive)
        (work/out.name).replace(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("short","background","commands"))
    stage = parser.parse_args().stage
    if stage == "commands":
        commands.main(OUT,COMMANDS,include_post120=True)
    else:
        replay(stage == "background")
