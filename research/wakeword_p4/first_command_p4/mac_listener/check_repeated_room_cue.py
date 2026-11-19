"""Score two fixed, repeated-marker WAVs and their playback-reference guards across 28 room responses."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import build_repeated_room_cue as source
import check_room_trained_cue as grid
import handoff_clock_rates as drift

ch, fit = grid.ch, source.fit
OUT = ch.RESULTS / (source.STEM+"-grid.json")


def tracks():
    plan = ch.read(source.PLAN)
    extra = {}
    for name,p in source.FILES.items():
        x,rate = sf.read(p,dtype="int16")
        assert rate==16000 and ch.sha(p)==plan["records"][name]["wav_sha256"]
        extra["repeat_"+name]=x
    return extra


def materials():
    extra = tracks()
    for group, ids in (("initial",fit.EVAL),("additional",source.EXTRA)):
        for s, signals, control in grid.scenes(ids,extra):
            yield group, s, {k:signals[k] for k in ("v5",*extra)}, control


def main():
    native = grid.room.rooms.c161_binding()
    old = ch.read(grid.OUT)
    by_scene = {(r["source"]["room_id"],r["source"]["kind"],r["source"]["level_db"],r["source"]["player_ppm"]):r for r in old["records"]}
    expected = ch.read(source.PLAN)
    assert list(source.EXTRA)==expected["extra_room_ids"]
    rows,controls=[],{}
    counts=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    dry={"v5":sf.read(fit.fixed.CUE,dtype="int16")[0],**tracks()}
    refs={}
    with TemporaryDirectory() as tmp:
        detector=fit.fixed.ch.binding(Path(tmp)/"v5.so",coded="v5")
        pair=fit.fixed.ch.binding(Path(tmp)/"pair.so",coded="aligned")
        for group,s,mixes,control in materials():
            key=(s["room_id"],s["kind"],s["level_db"],s["player_ppm"])
            h=ch.digest(control)
            if h not in controls:
                controls[h]=dict(group=group,source=s,result=grid.outcome(detector,native,control,s))
            results={}
            for name,x in mixes.items():
                r=grid.outcome(detector,native,x,s)
                if name=="v5" and group=="initial":
                    prior=by_scene[key]
                    assert r==prior["routes"]["v5"] and h==prior["control_sha256"]
                refkey=(name,s["start"],s["samples"],s["player_ppm"],s["level_db"])
                if refkey not in refs:
                    wave=drift.project(dry[name],grid.Fraction(1000000,1000000+s["player_ppm"]))
                    ref=np.zeros_like(x)
                    ref[s["start"]:s["command_at"]]=grid.room.quantize(wave.astype(float)*10**(s["level_db"]/20))
                    refs[refkey]=ref
                guard=drift.guarded(r,detector,pair,refs[refkey])
                r["self"]=guard
                results[name]=r
                for bucket in ("all",str(s["level_db"])):
                    c=counts[group][bucket][name]
                    c.update(trials=1,timely=r["timely"],opens=r["opened"] is not None,
                             self_open=bool(guard["opened"]),duplicate_fixed=len(r["fixed"])>1)
                    if name!="v5":
                        base=results["v5"]
                        c.update(gained=r["timely"] and not base["timely"], lost=base["timely"] and not r["timely"],
                                 earlier=r["timely"] and base["timely"] and r["opened"]<base["opened"])
            rows.append(dict(group=group,source=s,routes=results,control_sha256=h))
            if len(rows)%96==0: print(len(rows),flush=True)
    assert len(rows)==672 and len(controls)==56
    ch.save(OUT,dict(records=rows,controls=controls,summary=counts,
        sources={p.name:ch.sha(p) for p in (source.PLAN,grid.OUT,fit.FIT,fit.fixed.CUE,*source.FILES.values(),Path(__file__),Path(grid.__file__))},
        policy="Host streaming implementation with frozen C161, unmodified fixed v5 matcher, and original "
        "reference guard. A positive requires first OPEN after cue start and strictly before command. "
        "Self-play uses an aligned digital reference of the played wake at its corresponding level; "
        "background and command remain external. Repeated levels, rates and assets are paired observations."))


if __name__ == "__main__":
    main()
