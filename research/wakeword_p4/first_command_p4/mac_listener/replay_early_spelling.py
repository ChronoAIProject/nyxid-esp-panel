"""Replay calibrated early decisions, the original delayed decisions and single-worker timing."""
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

import fit_early_spelling as fit
import paired_piper_willow as matched
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_spelling_p4_commands import outcome

base,replay = fit.base,fit.replay


def groups():
    old = read(RESULTS/(fit.gap.STEM+"-comparison.json"))["records"]
    for g in fit.gap.study.GROUPS:
        timing = read(RESULTS/(replay.fit.music.STEM+"-"+g+".json"))["records"]
        for (idx,pcm,c161,meta,spans),t,r in zip(replay.sources(g),timing,old[g],strict=True):
            assert base.poll.digest(pcm)==t["pcm_sha256"]==r["pcm_sha256"] and r["index"]==idx
            yield g,idx,pcm,c161,meta,spans,t["proposals"],r["events"]["gap_and_five"]
    for g,name in (("joe","4090-c161-name-final-s-joe"),("dialogue","4090-c161-spelling-gap-dialogue-replay")):
        for (idx,pcm,c161,meta,spans),r in zip(replay.sources(g),read(RESULTS/(name+".json"))["records"],strict=True):
            assert base.poll.digest(pcm)==r["pcm_sha256"] and r["index"]==idx
            yield g,idx,pcm,c161,meta,spans,[c["q8"]["proposal"] for c in r["checks"]],r["events"]["gap_and_five"]
    plans=read(matched.PLAN)["records"]
    for r in read(RESULTS/(matched.STEM+"-proposals.json"))["records"]:
        idx=r["index"]
        s=plans[idx]
        pcm=replay.probe.dialogue.piper.load_wav(s)
        assert base.poll.digest(pcm)==s["pcm_sha256"]==r["mic_sha256"]
        meta=dict(cue_at=s["start"],command_at=s["end"],expected=s["carrier"],role=s["target_id"],cue=s["cue"])
        yield "paired",idx,pcm,[],meta,None,r["proposals"]["parent"],r["events"]["parent_five"]["events"]


def simulate(c161,ts,checks,length,choice,service):
    early=choice is not None
    jobs=[]
    for t in ts:
        if early and t+choice["delay"]<=length:
            jobs.append((t+choice["delay"],t,True))
        if t+fit.WAIT<=length:
            jobs.append((t+fit.WAIT,t,False))
    completed,count,early_accepted=0,0,set()
    events=[]
    for when,t,first in sorted(jobs):
        if not first and t in early_accepted:
            continue
        count+=1
        completed=max(when,completed)+service
        gate=checks[when]
        if gate["passes"] and (not first or gate["score"]>=choice["threshold"]):
            if first:
                early_accepted.add(t)
            events.append(completed)
    return dict(events=base.scoring.merged(c161,events),windows=count,early=len(early_accepted))


def main():
    config=read(fit.FIT)
    gates=fit.Gates()
    assert gates.ids==config["models"] and config["script_sha256"]==sha(Path(fit.__file__))
    selected=config["selected"]
    char_file=RESULTS/"4090-c161-spelling-p4.json"
    five_file=RESULTS/"4090-c161-positive-readout-p4.json"
    char_us = int(np.quantile([r["invoke_us"]+r["decode_us"] for r in read(char_file)["records"]],.95,method="higher"))
    five_us = int(np.quantile([r["invoke_us"] for r in read(five_file)["records"]],.95,method="higher"))
    service=((char_us+five_us)*16+999)//1000
    reader,identity=base.scoring.word.load_reader()
    counters=defaultdict(lambda:defaultdict(Counter))
    records=defaultdict(list)
    for g,idx,pcm,c161,meta,spans,proposals,baseline in groups():
        ts=[t for t in proposals if 0<t<=len(pcm)]
        ends=sorted({t+d for t in ts for d in (selected["delay"],fit.WAIT) if t+d<=len(pcm)})
        checks=gates.windows(base.native_inputs(gates.native,pcm)[0],ends) if ends else {}
        routes={k:simulate(c161,ts,checks,len(pcm),rule,cost) for k,rule,cost in
                (("baseline",None,0),("early",selected,0),("serial_baseline",None,service),("serial_early",selected,service))}
        assert routes["baseline"]["events"]==baseline,(g,idx)
        row=dict(index=idx,pcm_sha256=base.poll.digest(pcm),proposals=ts,checks=checks,routes=routes,source=meta)
        cache={}
        for k,result in routes.items():
            total=counters[g][k]
            ev=result["events"]
            total.update(dict(events=len(ev),windows=result["windows"],early=result["early"],samples=len(pcm)))
            if g in ("paired","matched","acoustics"):
                kind=meta.get("role",meta.get("acoustic"))
                r=outcome(ev,meta,pcm,reader,cache)
                result["scene"]=r
                total[kind+".trials"]+=1
                total[kind+".timely"]+=r["timely"]
                total[kind+".complete"]+=r.get("complete_suffix",False)
                total[kind+".hits"]+=bool(ev)
            elif spans is not None:
                replay.tally(ev,spans,g,total)
            elif g=="joe":
                total[meta["label"]+".trials"]+=1
                total[meta["label"]+".hits"]+=bool(ev)
        if cache:row["captures"]=cache
        records[g].append(row)
    save(RESULTS/(fit.STEM+"-replay.json"),dict(records=records,summary=counters,fit_sha256=sha(fit.FIT),
         asr=identity,char_p95_us=char_us,five_p95_us=five_us,service_samples=service,
         char_timing_sha256=sha(char_file),five_timing_sha256=sha(five_file),script_sha256=sha(Path(__file__)),
         policy="Preserve first-stage proposal clocks and independently admitted C161. One early check at the "
                "training-selected offset, otherwise the full 360 ms fallback when available. Baseline must match "
                "all prior events. Serial simulation charges the sum of the two separate QEMU P95 times for both "
                "classifiers on every evaluated window, no short circuit or frontend charges. Room and paired "
                "command completeness uses the historical two-second raw capture only when first event is timely."))


if __name__=="__main__":
    main()
