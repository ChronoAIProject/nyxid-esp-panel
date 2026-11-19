"""Measure equal-peak and equal-RMS waveforms on the identical microphone gains and backgrounds."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

import normalize_marker_gains as levels

study, ch = levels.study, levels.ch
grid, prior = study.grid, study.prior
OUT = ch.RESULTS/(study.fit.STEM+"-normalized.json")


def tracks():
    dry=study.tracks()
    plan=ch.read(levels.FIT)
    for name,wav in levels.WAV.items():
        q,rate=sf.read(wav,dtype="int16")
        assert rate==16000 and ch.sha(wav)==plan["records"][name]["wav_sha256"]
        dry[name]=q
    return dry


def main():
    dry=tracks()
    previous=ch.read(study.OUT)["records"]
    native=grid.room.rooms.c161_binding()
    rows=[]
    tally=defaultdict(lambda:defaultdict(Counter))
    with TemporaryDirectory() as tmp:
        full=ch.binding(Path(tmp)/"v5.so",coded="v5")
        pair=ch.binding(Path(tmp)/"pair.so",coded="aligned")
        for group,stream in (("established",study.established(dry)),("other_music",study.more_music(dry))):
            for i,(s,signals,control) in enumerate(stream):
                old=previous[len(rows)]
                assert old["source"]==s and ch.digest(control)==old["control_sha256"] and old["group"]==group
                for n,r in old["routes"].items():
                    assert ch.digest(signals[n])==r["pcm_sha256"]
                routes={}
                for name in levels.WAV:
                    result=grid.outcome(full,native,signals[name],s)
                    ref=prior.level.guard.refs(s,name,dry)
                    self_play=grid.drift.guarded(result,full,pair,ref)
                    routes[name]=dict(timely=result["timely"],opened=result["opened"],c161=result["c161"],
                                      fixed=result["fixed"],self_opened=self_play["opened"],pcm_sha256=result["pcm_sha256"],
                                      reference_sha256=self_play["reference_sha256"])
                    timely=result["timely"]
                    base=old["routes"]["v5"]
                    tally[group][name].update(trials=1,timely=timely,gained=timely and not base["timely"],
                                              lost=base["timely"] and not timely,self_open=bool(self_play["opened"]))
                rows.append(dict(group=group,index=i,routes=routes))
    assert len(rows)==len(previous)==1288
    ch.save(OUT,dict(records=rows,summary=tally,sources={p.name:ch.sha(p) for p in (study.OUT,levels.FIT,Path(__file__))},
                     policy="Same baseline/training identities and case order as previous report. Equal-peak/RMS "
                     "wet outputs share exact microphone gain, checked against original control and three source hashes."))


if __name__=="__main__":
    main()
