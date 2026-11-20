"""Add verified subthreshold proposals to the original single C161 route and replay continuously."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import replay_c161_event_patch as replay
from replay_c161_history_gain_cohorts_v1 import REFERENCE
from replay_c161_vowel_name_pair_v1 import cases

fit,base=replay.fit,replay.base
OUT=base.RESULTS/(fit.STEM+"-rescue.json")


def combined(sums,clock,eligible,scores,models):
    accepted=np.ones(len(sums),bool)
    result=dict(baseline=replay.events(sums,clock,1175,accepted))
    for arm,(_,cut) in models.items():
        effective=sums.copy()
        passed=eligible[np.asarray(scores[arm],dtype=np.int64)>=cut]
        effective[passed]=np.maximum(effective[passed],1175)
        result[arm]=replay.events(effective,clock,1175,accepted)
    return result


def infer(bands,eligible,models):
    x=fit.patches(bands,eligible) if len(eligible) else np.empty((0,120),np.uint8)
    return {arm:fit.integer_scores(x,params).tolist() for arm,(params,_) in models.items()}


def backgrounds(native,models):
    src=base.read(REFERENCE)
    rows=[]
    for cohort in ("primed","jenny","melo"):
        path=Path(src["cohorts"][cohort]["path"])
        assert base.sha(path)==src["cohorts"][cohort]["sha256"]
        for r in base.read(path)["records"]:
            if r["phase"]!="background":
                continue
            source=r["source"]
            trace=Path(r["trace"]["path"])
            assert base.sha(trace)==r["trace"]["sha256"]
            with np.load(trace) as z:
                codes=z["c161_uint8"]
                clock=z["relative_block_return"]
            sums=np.convolve(codes.astype(np.int32),np.ones(5,np.int32),"full")[:len(clock)]
            eligible=np.flatnonzero((clock>0)&(sums>=960))
            if len(eligible):
                pcm=replay.material.wave(source)
                # Keep the original frontend history and include the complete final selected input block.
                stop=min(len(pcm),int(clock[eligible[-1]])+480)
                _,bands,short_clock=fit.replay.native_inputs(native,pcm[:stop])
                np.testing.assert_array_equal(short_clock[eligible],clock[eligible])
                scores=infer(bands,eligible,models)
                events=combined(sums,clock,eligible,scores,models)
            else:
                scores={arm:[] for arm in models}
                events={arm:[] for arm in ("baseline",*models)}
            assert not events["baseline"] and not r["events"]["c161"]
            rows.append(dict(cohort=cohort,index=r["index"],source=source.get("source_id") or source["id"],pcm_sha256=source["pcm_sha256"],
                             samples=source["samples"],trace_sha256=r["trace"]["sha256"],peak=int(sums.max()),
                             proposed_at=clock[eligible].tolist(),sums=sums[eligible].tolist(),scores=scores,events=events))
    return rows


def main():
    training=base.read(fit.FIT)
    assert base.sha(base.MODEL)==training["model_sha256"]
    models={}
    for arm,spec in training["models"].items():
        path=base.RESULTS/spec["weights"]
        assert base.sha(path)==spec["sha256"]
        with np.load(path) as z:
            models[arm]=dict(z),spec["cut"]
    native=fit.native_frontend()
    development=base.read(replay.OUT)
    rows=[]
    counts=defaultdict(lambda:defaultdict(Counter))
    for old,(meta,pcm,_) in zip(development["records"],replay.inputs(),strict=True):
        assert old["pcm_sha256"]==replay.sha256(pcm.tobytes()).hexdigest()
        _,clock,codes,sums=fit.stream(native,pcm)
        assert replay.sha256(codes.tobytes()).hexdigest()==old["codes_sha256"]
        eligible=np.flatnonzero((clock>0)&(sums>=960))
        assert clock[eligible].tolist()==old["check_at"]
        hits=combined(sums,clock,eligible,old["scores"],models)
        assert hits["baseline"]==old["events"]["raw_1175"]
        for arm,ts in hits.items():
            baseline=bool(hits["baseline"])
            count=counts[meta["cohort"]+"."+meta["role"]][arm]
            count.update(clips=1,opened=int(bool(ts)),gained=int(bool(ts) and not baseline),lost=int(baseline and not ts))
            if "command_at" in meta:
                count.update(before_command=int(any(meta["cue_start"]<=((t+255)//256)*256<meta["command_at"] for t in ts)))
        rows.append(dict(meta,pcm_sha256=old["pcm_sha256"],events=hits))
    sessions=[]
    for meta,pcm in cases("context"):
        if "timeline" not in meta:
            continue
        bands,clock,_,sums=fit.stream(native,pcm)
        eligible=np.flatnonzero((clock>0)&(sums>=960))
        hits=combined(sums,clock,eligible,infer(bands,eligible,models),models)
        spans=[]
        for s in meta["timeline"]:
            if s["kind"]=="clip":
                spans.append(dict(text=s["text"],group=s["group"],start=s["start"],end=s["end"],
                                  events={arm:[t for t in ts if s["start"]<t<=s["end"]] for arm,ts in hits.items()}))
        sessions.append(dict(key=meta["key"],pcm_sha256=replay.sha256(pcm.tobytes()).hexdigest(),events=hits,spans=spans))
    background=backgrounds(native,models)
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        path=Path(tmp)/OUT.name
        base.save(path,dict(records=rows,summary=counts,sessions=sessions,background=background,
                    sources={p.name:base.sha(p) for p in (Path(__file__),replay.OUT,fit.FIT,REFERENCE)},
                    policy="Any raw five-score sum >=1175 always qualifies. Between 960 and 1174 a positive integer "
                           "check also qualifies. Original shared 67-step cooldown and re-arm below half of 1175. "
                           "Use each full input clock, including checks after a rejected proposal. Backgrounds use "
                           "their original authenticated raw C161 traces and the complete PCM prefix up to each "
                           "qualifying window. Historical sets stay separate."))
        path.replace(OUT)
    print(len(rows),len(sessions),len(background),flush=True)


if __name__=="__main__":
    main()
