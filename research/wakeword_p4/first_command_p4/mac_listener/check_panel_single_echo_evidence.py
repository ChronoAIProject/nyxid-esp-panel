"""Compare causal, time-local single-C corroboration with the acoustic B,C pair on the same raw PCM."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import check_panel_echo_confirmation as pair
from diagnose_c161_v5_room_misses import correlations

tx, ch = pair.tx, pair.ch
OUT = ch.RESULTS/"4090-c161-panel-single-echo-evidence.json"
CONNECTED = ch.RESULTS/"4090-c161-connected-nick-willow-plan.json"


def hits(x, event):
    if not event:
        return []
    # Preserve 4-sample frontend phase, 17 FIR taps and the full 320-tick
    # code history before the first possible match.
    begin = max(0,(event-pair.SKEW-4*(320+17))//4*4)
    end=min(len(x),event+pair.SKEW)
    score=correlations(x[begin:end])[2]
    clock=begin+4*(1+np.arange(len(score)))
    return clock[(np.abs(score)>=0.30)&(abs(clock-event)<=pair.SKEW)].tolist()


def main():
    previous=ch.read(pair.OUT)
    baseline=ch.read(tx.OUT)
    rows,rooms,spoken,names,stats=[],[],[],[],defaultdict(Counter)
    def score(group,x,raw,old):
        assert ch.digest(x)==raw["pcm_sha256"]
        event=old["pair"]
        ref=np.zeros_like(x)
        single=hits(x,event)
        new=pair.admit(raw,event,single,ref)["opened"]
        stats[group].update(trials=1, unprotected=int(bool(old["unprotected"])),timer=int(bool(old["timer"])),
                            bc=int(bool(old["confirmed"])), single=int(bool(new)))
        return dict(tx=event,c_hits=single,bc_open=old["confirmed"],single_open=new,pcm_sha256=raw["pcm_sha256"])
    old_rows={r["host_index"]:r for r in previous["records"]}
    raw_rows={r["host_index"]:r for r in baseline["records"]}
    for i,(s,n,role,x,_) in enumerate(tx.study.simultaneous()):
        if i in old_rows:
            rows.append(dict(index=i,role=role,**score("overlap."+role,x,raw_rows[i]["raw"],old_rows[i])))
    for i,((s,x,raw,_),old) in enumerate(zip(pair.room_trials(),previous["rooms"],strict=True)):
        rooms.append(dict(index=i,**score("room."+s["kind"],x,raw,old)))
    for i,((_,x),old) in enumerate(zip(tx.spoken_inputs(),previous["spoken"],strict=True)):
        spoken.append(dict(index=i,**score("isolated_spoken",x,baseline["spoken"][i]["raw"],old)))
    with TemporaryDirectory() as folder:
        matcher=tx.single.timed.binding(Path(folder))
        native=tx.panel.grid.room.rooms.c161_binding()
        echoed=tx.study.reference_binding(Path(folder)/"echo")
        for s in ch.read(CONNECTED)["records"]:
            if s["target_id"] not in ("Hey Nyx","Hey Nicks"):
                continue
            x,sr=sf.read(s["path"],dtype="int16")
            assert sr==16000 and ch.digest(x)==s["pcm_sha256"]
            raw=tx.panel.grid.fit.fixed.ch.routes(matcher,native,x)
            # Calibrate the virtual clock to the observed hit for a worst-case
            # isolated overlap. That clock is an experimental stress condition.
            for lag in (0,1600,3200,4800):
                event=raw["c161"][0]-lag if raw["c161"] else s["end"]-lag
                zero=np.zeros_like(x)
                bc=[h["t"] for h in tx.study.host.detect(echoed,[x])]
                old=dict(pair=event,unprotected=tx.apply(raw,0,zero)["opened"],
                         timer=tx.apply(raw,event,zero)["opened"],confirmed=pair.admit(raw,event,bc,zero)["opened"])
                result=score("connected."+s["target_id"],x,raw,old)
                names.append(dict(index=s["index"],voice=s["voice"],target=s["target_id"],event_to_hit=lag,
                                  raw_hits=raw["c161"],**result))
    ch.save(OUT,dict(records=rows,rooms=rooms,isolated=spoken,connected=names,summary=stats,
        sources={p.name:ch.sha(p) for p in (pair.OUT,tx.OUT,CONNECTED,Path(__file__))},
        policy="Host GPU float64 diagnosis only. Use the original 0.30 signed-magnitude threshold on template C, "
               "within 80 ms of the panel playout clock. Require both observations before veto. The two residual "
               "development misses motivated this comparison. Report ordinary and connected overlap conditions."))


if __name__=="__main__":
    main()
