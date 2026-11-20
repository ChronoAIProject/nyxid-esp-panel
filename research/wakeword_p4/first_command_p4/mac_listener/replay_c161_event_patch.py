"""Replay the frozen acoustic checks on natural targets, common phrases and the board false wake."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
import c161_binary40_parallel_sources_v1 as material
import check_six_cups_clock as six
import connected_nick_willow as connected
import train_c161_event_patch as fit

base=fit.base
OUT=base.RESULTS/(fit.STEM+"-development.json")


def events(sums,clock,fence,passed):
    armed=True
    last=-1000
    hits=[]
    for i in range(4,len(clock)):
        if i-last<67:
            continue
        if 2*sums[i]<fence:
            armed=True
        if armed and sums[i]>=fence and passed[i]:
            hits.append(int(clock[i]))
            armed=False
            last=i
    return hits


def evaluate(native,models,pcm):
    bands,clock,codes,sums=fit.stream(native,pcm)
    eligible=np.flatnonzero((clock>0)&(sums>=960))
    observations=fit.patches(bands,eligible) if len(eligible) else np.empty((0,120),np.uint8)
    scores={}
    controls=np.ones(len(clock),dtype=bool)
    result={f"raw_{f}":events(sums,clock,f,controls) for f in (960,1175)}
    assert result["raw_1175"]==times(decode(codes,clock))
    for arm,(params,cut) in models.items():
        logits=fit.integer_scores(observations,params)
        passed=np.zeros(len(clock),bool)
        passed[eligible]=logits>=cut
        scores[arm]=logits.tolist()
        for f in (960,1175):
            result[f"{arm}_{f}"]=events(sums,clock,f,passed)
    return dict(events=result,check_at=clock[eligible].tolist(),sums=sums[eligible].tolist(),scores=scores,
                codes_sha256=sha256(codes.tobytes()).hexdigest())


def history():
    for cohort in ("primed","jenny","joe","builtin","mac","daily"):
        old=base.read(base.RESULTS/f"4090-c161-ambient-history-{cohort}-v1.json")["records"]
        if cohort in ("primed","jenny","joe"):
            receipt=material.read(material.introot(cohort)/"result.json")
            rows=material.read(receipt["models"][0]["path"])["records"]
            samples=((r["index"],r["record"],material.wave(r["record"])) for r in rows if r["phase"]=="synthetic")
        else:
            samples=((i,r,pcm) for i,r,pcm,*_ in material.inputs(cohort))
        for i,r,pcm in samples:
            assert old[i]["index"]==i
            role=r.get("target_id") or r.get("target") or r.get("group")
            yield dict(cohort=cohort,id=str(i),role=role,text=r["text"],voice=r.get("voice",cohort),
                       target=role in ("nyx","hey_nyx"),source_sha256=r.get("sha256") or r.get("wav_sha256")),pcm,times(old[i]["c161"])


def inputs():
    yield from history()
    source=base.read(connected.PLAN)
    assert base.sha(connected.PACK)==source["audio_sha256"]
    with ZipFile(connected.PACK) as z:
        for r in source["records"]:
            pcm=six.cue.gains.study.playback.read_wav(z,dict(file=r["member"],sha256=r["sha256"],pcm_sha256=r["pcm_sha256"]))
            yield dict(cohort="connected",role=r["target_id"],id=str(r["index"]),text=r["synthesis"]["text"],
                       target=r["target_id"] in ("Hey Nyx","Hey Nicks","fixed-v5"),voice=r["voice"],
                       cue_start=r["start"],command_at=r["end"]),pcm,None
    # The standalone negative WAV gives the length of the historical P4 digital microphone input.
    board=base.read(six.OUT)
    ref,rate=sf.read(six.WAV,dtype="int16")
    assert rate==16000 and base.sha(six.WAV)==board["wav_sha256"]
    seeds={key:pcm for s,key,_,pcm in six.cue.overlap.event_inputs("daily") if s["text"]==board["phrase"]}
    for r in board["records"]:
        pcm=np.zeros_like(ref)
        speech=seeds[r["speech_key"]]
        pcm[r["start"]:r["start"]+len(speech)]=speech
        assert sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
        yield dict(cohort="six_cups",role="conversation",id=r["label"],voice=r["voice"],text=board["phrase"],target=False),pcm,r["events"]["M"]


def main():
    report=base.read(fit.FIT)
    assert base.sha(base.MODEL)==report["model_sha256"]==base.MODEL_SHA
    models={}
    for arm,spec in report["models"].items():
        path=base.RESULTS/spec["weights"]
        assert base.sha(path)==spec["sha256"]
        with np.load(path) as z:
            models[arm]=dict(z),spec["cut"]
    native=fit.native_frontend()
    training={sha256(base.audio(r).tobytes()).hexdigest() for r in report["records"]}
    rows=[]
    counts=defaultdict(lambda:defaultdict(Counter))
    for meta,pcm,expected in inputs():
        pcm_sha=sha256(pcm.tobytes()).hexdigest()
        assert pcm_sha not in training
        result=evaluate(native,models,pcm)
        if expected is not None:
            assert result["events"]["raw_1175"]==expected,(meta,expected,result)
        row=dict(**meta,pcm_sha256=pcm_sha,**result)
        rows.append(row)
        original=bool(result["events"]["raw_1175"])
        for key,hits in result["events"].items():
            count=counts[meta["cohort"]+"."+meta["role"]][key]
            count.update(clips=1,opened=int(bool(hits)),gained=int(bool(hits) and not original),
                         lost=int(original and not hits))
            if "command_at" in meta:
                count.update(before_command=int(any(meta["cue_start"]<=((t+255)//256)*256<meta["command_at"] for t in hits)))
    with TemporaryDirectory(dir=fit.WORK) as tmp:
        path=Path(tmp)/OUT.name
        base.save(path,dict(records=rows,summary=counts,sources={p.name:base.sha(p) for p in (
                   Path(__file__),fit.FIT,six.OUT,connected.PLAN,connected.PACK)},model_sha256=base.MODEL_SHA,
                   policy="Host replay of every synthetic clip in six consumed cohorts, all matched connected "
                          "utterances and the 36 P4 negative clock tests. Check raw C161 events against the "
                          "archived clocks wherever available. A rejected check leaves the detector armed. "
                          "All arms retain 67-step cooldown and half-level rearm; evaluate both the original "
                          "five-code sum 1175 and the training crossing 960. Score the acoustic model route only."))
        path.replace(OUT)
    print(len(rows),"replays",flush=True)


if __name__=="__main__":
    main()
