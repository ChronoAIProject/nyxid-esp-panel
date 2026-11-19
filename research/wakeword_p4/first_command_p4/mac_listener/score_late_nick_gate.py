"""Measure trained Nick vetoes against full previous opening clocks and matched captures."""
from collections import Counter, defaultdict
from pathlib import Path

import fit_late_nick_gate as fit
import score_synthesized_call_context as ctx
import score_early_gate_voices_opus as audio

ch,timing=fit.ch,fit.timing
OUT=ch.RESULTS/"4090-c161-late-nick-gate-replay.json"


def evaluate(row, checks, spec, heads, costs, service=None, rounded=True):
    windows,features={},{}
    checks={int(t):w for t,w in checks.items()}
    for t,w in checks.items():
        us=service[t] if service is not None else costs["char_us"]+(costs["five_us"] if w["accepted"] else 0)
        passed=w["accepted"] and w["five"][0]>=spec["models"]["cutoff"]
        if w["accepted"]:
            features[t]=fit.voices.gate.fit.features(w,w["five"])
        windows[t]=dict(id=t,service_us=us,passes=passed)
    def events(w):
        ts=fit.voices.gate.routes(row,w,features,spec,0)["linear"]["events"]
        return [timing.block(t) for t in ts] if rounded else ts
    baseline=events(windows)
    results={"prior":baseline}
    for name,head in heads.items():
        updated=dict(windows)
        for t in row["stages"]["filtered"]["proposals"]:
            end=t+timing.STANDARD
            if end not in updated or not updated[end]["passes"]:
                continue
            x=fit.features(checks[t+timing.EARLY],checks[end])[head["offset"]:]
            value=sum(a*b for a,b in zip(x,head["weights"],strict=True))
            assert -(1<<31)<=value<1<<31
            updated[end]=dict(windows[end],passes=value>=head["cutoff"])
        results[name]=events(updated)
    return results


def main():
    spec=ch.read(fit.voices.gate.fit.COEFF)
    train=ch.read(fit.FIT)
    assert spec["models"]==train["models"]
    parent=ch.read(timing.names.OUT)
    all_rows=[]
    summary=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    def record(group,index,role,routes,expected,clock,spans=None,captures=None):
        assert routes["prior"]==expected,(group,index,routes["prior"],expected)
        item=dict(group=group,index=index,role=role,routes=routes,capture={})
        for k,ts in routes.items():
            total=summary[group][role][k]
            hits=[t for t in ts if clock["start"]<=t<clock["end"]]
            total.update(trials=1,hits=bool(hits),events=len(ts))
            if clock["command_at"]:
                total["timely"]+=bool(hits and hits[0]<clock["command_at"])
            if captures is not None and hits:
                if hits[0] in captures:
                    c=captures[hits[0]]
                    total.update(complete=c,timely_complete=c and hits[0]<clock["command_at"])
                    item["capture"][k]=c
                else:
                    total["unverified_capture"]+=1
            if spans is not None:
                fit.voices.study.replay.tally(ts,spans,group,total)
        all_rows.append(item)

    banks=ch.read(audio.OUT)
    capmap={(r["group"],r["index"]):{v["opened"]:v["complete"] for v in r["routes"].values() if v is not None}
            for r in banks["records"]}
    allvoices=ch.read(fit.voices.OUT)
    for group,rows in allvoices["records"].items():
        for r in rows:
            meta=r["source"]
            clock=dict(start=meta.get("cue_at",0),end=r["samples"],command_at=meta.get("command_at",0),
                       stages=dict(raw=dict(c161=r["c161"]),filtered=dict(c161=[],proposals=r["proposals"])))
            routes=evaluate(clock,r["checks"],spec,train["heads"],parent)
            role=meta.get("role",meta.get("label",meta.get("acoustic",group)))
            record(group,r["index"],role,routes,r["routes"]["linear"],clock,r["spans"],capmap.get((group,r["index"])))
    interference=ch.read(fit.voices.gate.interference.OUT)
    calibrated=ch.read(fit.voices.gate.OUT)
    for group,rows in interference["records"].items():
        for r,ref in zip(rows,calibrated["interference"][group],strict=True):
            assert r["pcm_sha256"]==ref["pcm_sha256"]
            clock=dict(start=0,end=r["samples"],command_at=0,
                       stages=dict(raw=dict(c161=r["independent"]),filtered=dict(c161=[],proposals=r["proposals"])))
            runs=evaluate(clock,r["checks"],spec,train["heads"],parent,rounded=False)
            spans=r["source"]["source"]["spans"] if group in ("bot","daily") else None
            record(group,r["index"],group,runs,ref["routes"]["linear"],clock,spans)
    lengths=[s["samples"] for s in ch.read(ctx.challenge.PLAN)["records"]]
    context=ch.read(ctx.OUT)
    for r in context["records"]:
        expected=r["routes"]["linear"]["events"]
        # The input length is defined by the prepublished waveform plan.
        clock=dict(start=r["cue_at"],end=lengths[r["index"]],command_at=r["command_at"],
                   stages=dict(raw=dict(c161=r["c161"]),filtered=dict(c161=[],proposals=r["proposals"])))
        caps={v["opened"]:v["complete"] for v in r["routes"].values() if v["events"]}
        runs=evaluate(clock,r["checks"],spec,train["heads"],parent)
        record("context_"+r["style"],r["index"],r["role"],runs,expected,clock,captures=caps)
    char,five=ch.read(timing.CHAR),ch.read(timing.FIVE)
    device={w["character_window"]:w for w in five["records"]}
    residual_audio=ch.read(fit.voices.gate.opus.OUT)
    cap_by_ordinal={r["ordinal"]:r for r in residual_audio["records"]}
    for i,(r,s,ref) in enumerate(zip(parent["records"],char["streams"],calibrated["records"],strict=True)):
        assert s["pcm_sha256"]==r["stages"]["filtered"]["pcm_sha256"]
        obs,service={},{}
        for j in s["windows"]:
            w=char["records"][j]
            t=w["end"]
            q=dict(w["device"],accepted=j in device)
            us=w["invoke_us"]+w["decode_us"]
            if j in device:
                q["five"]=device[j]["device"]
                us+=device[j]["invoke_us"]
            obs[t]=q
            service[t]=us
        runs=evaluate(r,obs,spec,train["heads"],parent,service,rounded=False)
        # Parent calibration leaves the residual C161 branch muted before its cutoff.
        assert all(t>=parent["cutoff_samples"] for t in r["stages"]["filtered"]["c161"])
        caps=None
        if i in cap_by_ordinal:
            caps={v["clock"]:residual_audio["captures"][v["capture"]]["lanes"]["residual"]["complete"]
                  for v in cap_by_ordinal[i]["routes"].values() if v is not None}
        record("calibrated_echo",i,r["role"],runs,ref["routes"]["linear"]["events"],r,captures=caps)
    sources=(fit.FIT,fit.voices.gate.fit.COEFF,fit.voices.OUT,audio.OUT,ctx.OUT,ctx.challenge.PLAN,
             fit.voices.gate.interference.OUT,fit.voices.gate.OUT,timing.CHAR,timing.FIVE,timing.names.OUT,
             fit.voices.gate.opus.OUT,Path(__file__))
    ch.save(OUT,dict(records=all_rows,summary=summary,sources={p.name:ch.sha(p) for p in sources},
         policy="Only change the 360 ms fallback timing gate. Compare each stream to its archived integer-early "
         "route and use the original per-window service times. Reuse a decoded capture only for a matching "
         "source and first event. Preserve each cohort's timing convention and report missing captures."))


if __name__=="__main__":
    main()
