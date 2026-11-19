"""Count unprompted C161 events on each complete development music track, from a fresh state."""
from pathlib import Path

import numpy as np
import soundfile as sf

import check_spoken_wake_eq as study
from fit_single_marker_gains import DEV

ch = study.ch
OUT = ch.RESULTS/(study.fit.STEM+"-music.json")


def main():
    historical = ch.RESULTS/"4090-c161-music-development-v1.json"
    originals = ch.read(historical)
    spec = ch.read(study.fit.FIT)
    coeff = np.asarray(spec["taps"][spec["selected"]])
    native = study.room.rooms.c161_binding()
    rows = []
    for source, old in zip(ch.read(DEV)["records"],originals["records"],strict=True):
        assert ch.sha(Path(source["path"]))==source["sha256"] and source["source_path"]==old["source_path"]
        x,sr=sf.read(source["path"],dtype="int16")
        assert sr==16000 and ch.digest(x)==source["pcm_sha256"]==old["pcm_sha256"]
        y,clips=study.fit.process(x,coeff)
        events,raw_scores=study.diagnostic.trace(native,x)
        filtered,eq_scores=study.diagnostic.trace(native,y)
        assert events==old["events"]["c161"]
        rows.append(dict(track=source["source_path"],samples=len(x),clipped=clips,pcm_sha256=ch.digest(x),
                         eq_sha256=ch.digest(y),raw=events,eq=filtered,raw_codes_sha256=raw_scores,
                         eq_codes_sha256=eq_scores,union=sorted(set(events+filtered))))
    ch.save(OUT,dict(records=rows,seconds=sum(r["samples"] for r in rows)/16000,
                     totals={key:sum(len(r[key]) for r in rows) for key in ("raw","eq","union")},
                     sources={p.name:ch.sha(p) for p in (DEV,historical,study.fit.FIT,Path(__file__))},
                     policy="Full tracks, continuous C161 state within each track. Previously consumed music "
                     "exposure, counted once. No synthetic wake or command mixed in."))


if __name__=="__main__":
    main()
