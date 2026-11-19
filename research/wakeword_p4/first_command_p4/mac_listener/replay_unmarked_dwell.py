"""Replay training-calibrated auxiliary dwell rules against full saved scores and real microphone clocks."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import calibrate_unmarked_dwell as fit
import post120_background as background

ch, replay, score = fit.ch, fit.replay, fit.replay.score
OUT = fit.ROOT / (fit.STEM+"-replay.json")


def materials(short, long):
    connected = iter(replay.recordings())
    ordinary = {g:iter(score.base.sources(g)) for g in ("daily","bot")}
    with np.load(replay.COMMON.with_suffix(".npz"),allow_pickle=False) as common:
        common_meta = iter(ch.read(replay.COMMON)["records"])
        for r in short["records"]:
            raw = echo = None
            if "cohort" in r:
                cohort, track, src, pcm, raw, echo, _ = next(connected)
                assert (r["cohort"],r["track"],r["source"])==(cohort,track,src)
            elif r["group"]=="common":
                s = next(common_meta)
                pcm = np.pad(common[s["key"]],(32000,16000))
                assert (s["voice"],s["text"],s["speed"])==(r["voice"],r["text"],r["speed"])
            else:
                meta,pcm,base,_,_ = next(ordinary[r["group"]])
                assert meta["source"]==r["source"] and base==r["events"]["baseline"]
            yield r,pcm,raw,echo
        for it in (connected,common_meta,*ordinary.values()):
            assert next(it,None) is None
    for r,(s,pcm,c161,clock_sha,_) in zip(long["records"],score.base.sources("background"),strict=True):
        assert s["key"]==r["key"] and not c161 and clock_sha==r["clock_sha256"]
        yield dict(r,group="background"),pcm,None,None


def main():
    calibrated, short, long = map(ch.read,(fit.OUT,replay.OUT,background.OUT))
    assert calibrated["models"]==short["models"]==long["models"]
    assert all(ch.sha(p.with_suffix(".npz"))==d["archive_sha256"] for p,d in ((replay.OUT,short),(background.OUT,long)))
    native = score.fit.prior.letters.native_frontend()
    five,five_ids = score.verifier.models()
    assert five_ids["trained"] == short["five_sha256"]
    arms = calibrated["policies"]
    stats, rows = defaultdict(Counter), []
    with np.load(replay.OUT.with_suffix(".npz"),allow_pickle=False) as short_z, np.load(background.OUT.with_suffix(".npz"),allow_pickle=False) as long_z:
        for r,pcm,raw,echo in materials(short,long):
            bg = r.get("group")=="background"
            z = long_z if bg else short_z
            key = r["key"]
            assert ch.digest(pcm)==r["pcm_sha256"] and len(pcm)==r["samples"]
            clock, codes = z[key+"_clock"], z[key+"_command"]
            old = r["events"] if bg else r["proposals"]
            assert fit.detect(codes,clock,5,1175)==old["command"]
            proposals = {name:fit.detect(codes,clock,p["width"],p["threshold"]) for name,p in arms.items()}
            checks = {int(k):v for k,v in r["name_scores"].items()}
            missing = sorted(set(t for hits in proposals.values() for t in hits)-checks.keys())
            if missing:
                bands,_,new_clock = score.base.native_inputs(native,pcm)
                np.testing.assert_array_equal(new_clock,clock)
                for t in missing:
                    # A file ending mid-hop can return its exact length. The verifier only reads completed hops.
                    x = score.verifier.training.room.window(bands,t//160*160)
                    checks[t] = five["trained"].invoke(np.rint(x.clip(0,666)*(255/666)).astype(np.uint8)).tolist()
            res = dict(key=key,pcm_sha256=r["pcm_sha256"],proposals=proposals,name_scores=checks)
            for name,hits in list(proposals.items()):
                proposals[name+"_confirmed"] = [t for t in hits if checks[t][0]>=short["cutoff"]]
            if raw is not None:
                s = r["source"]
                opened = {}
                for name,hits in proposals.items():
                    combined = dict(raw,c161=sorted(set(raw["c161"]+hits)))
                    ts = score.mixed.panel.middle.pair.admit(combined,s["transmit_event"],echo,np.zeros_like(pcm))["opened"]
                    g = stats[".".join((r["cohort"],r["track"],name))]
                    score.mixed.study.tally(g,s,dict(opened=ts))
                    g.update(by_150ms=int(bool(ts and s["start"]<=ts[0]<=s["command_at"]+2400)))
                    opened[name] = ts
                diagnostic = {}
                for model in ("original","command"):
                    trace = z[key+"_"+model]
                    diag = {}
                    for width in fit.WIDTHS:
                        t = (clock[width-1:]+255)//256*256
                        mask = (s["start"]<=t)&(t<=s["command_at"]+2400)
                        levels = fit.sums(trace,width)
                        j = int(np.argmax(np.where(mask,levels,-1)))
                        diag[str(width)] = dict(peak=int(levels[j]),at=int(t[j]))
                    diagnostic[model] = diag
                res.update(cohort=r["cohort"],track=r["track"],source=s,opened=opened,peaks=diagnostic)
            elif bg:
                res.update(group="background",samples=len(pcm))
                for name,hits in proposals.items():
                    stats["background."+name].update(events=len(hits))
            else:
                group = r["group"]
                events = {name:score.base.scoring.merged(r["events"]["baseline"],hits) for name,hits in proposals.items()}
                res.update(group=group,events=events)
                for name,ts in events.items():
                    if group=="common":
                        stats[".".join((group,r["text"],name))].update(trials=1,triggered=int(bool(ts)))
                    else:
                        used=set()
                        for span in r["source"]["spans"]:
                            hits = [t for t in ts if span["start"]<=t<span["end"]+6400]
                            stats[".".join((group,span["address"],name))].update(trials=1,triggered=int(bool(hits)))
                            used.update(hits)
                        stats[".".join((group,"outside",name))].update(events=len(set(ts)-used))
            rows.append(res)
    assert len(rows)==374
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        path = Path(tmp)/OUT.name
        ch.save(path,dict(records=rows,summary=stats,models=short["models"],selected=calibrated["selected"],policies=arms,
                c161=short["c161"],five_sha256=short["five_sha256"],cutoff=short["cutoff"],background_samples=long["total_samples"],
                sources={p.name:ch.sha(p) for p in (fit.OUT,replay.OUT,background.OUT,Path(__file__))},
                policy="All three training-calibrated operating points reported. Source arrays and clocks checked. "
                       "Same confirmed self-play veto, zero reference and block delivery. Background is reused."))
        path.replace(OUT)


if __name__=="__main__":
    main()
