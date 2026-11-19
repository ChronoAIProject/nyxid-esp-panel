"""Pair an ordinary synthetic Nyx and a real acoustic panel demonstration on the established scene clocks."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_panel_single_echo_evidence as single

pair,tx,ch=single.pair,single.tx,single.ch
OUT=ch.RESULTS/"4090-c161-panel-spoken-barge-in.json"


def main():
    spoken,sr=sf.read(tx.SPOKEN,dtype="int16")
    assert sr==16000
    e=tx.event_time()
    records=[]
    totals=defaultdict(Counter)
    source=ch.read(tx.diag.BOARD)
    with TemporaryDirectory() as folder:
        work=Path(folder)
        matcher=tx.single.timed.binding(work)
        native=tx.panel.grid.room.rooms.c161_binding()
        bc=tx.study.reference_binding(work/"pair")
        for row in source["records"]:
            if (row["role"],row["name"])!=("both","bca"): continue
            s=row["source"]
            _,noise,local,command,h,gain,_=tx.diag.scene(row)
            at=s["start"]
            local_at=s["local_at"]
            end=at+len(spoken)
            external=fftconvolve(np.r_[spoken,command].astype(float),h)*gain
            cmd=fftconvolve(command.astype(float),h)*gain
            base=noise.copy()
            base[local_at:local_at+len(local)]+=local
            mixed={}
            for name,base_wave,src,start in (
                ("external_only",noise,external,at),("both",base,external,at),("panel_plus_command",base,cmd,end)):
                mic=base_wave.copy()
                mic[start:start+len(src)]+=src
                mic=tx.study.grid.room.quantize(mic*s["gain"])
                zero=np.zeros_like(mic)
                raw=tx.panel.grid.fit.fixed.ch.routes(matcher,native,mic)
                clock=0 if name=="external_only" else local_at+e
                bc_hits=[v["t"] for v in tx.study.host.detect(bc,[mic])]
                c_hits=single.hits(mic,clock)
                opens=dict(unprotected=tx.apply(raw,0,zero)["opened"],timer=tx.apply(raw,clock,zero)["opened"],
                    pair=pair.admit(raw,clock,bc_hits,zero)["opened"],middle=pair.admit(raw,clock,c_hits,zero)["opened"])
                mixed[name]=dict(raw=raw,pair_hits=bc_hits,c_hits=c_hits,clocks=opens)
                for mode,cl in opens.items():
                    totals[name+"."+mode].update(trials=1,opens=int(bool(cl)),
                        timely=int(bool(cl and at<=cl[0]<end)),early=int(bool(cl and cl[0]<at)))
            for mode in mixed["both"]["clocks"]:
                alone=mixed["external_only"]["clocks"][mode]
                both=mixed["both"]["clocks"][mode]
                neg=mixed["panel_plus_command"]["clocks"][mode]
                totals["paired."+mode].update(trials=1,external_only_hits=int(bool(alone)),
                    retained_with_panel=int(bool(alone and both and both[0]>=at and not neg)),
                    lost_with_panel=int(bool(alone and not both)),
                    without_address_open=int(bool(neg)))
            records.append(dict(scene=row["index"],geometry=s["geometry"],offset=s["offset"],external_db=s["external_db"],
                           voice=s["voice"],command=s["command"],external_at=at,command_at=end,views=mixed))
    assert len(records)==60
    ch.save(OUT,dict(records=records,summary=totals,sources={p.name:ch.sha(p) for p in
        (tx.diag.BOARD,tx.SPOKEN,single.OUT,Path(__file__))},
        policy="Acoustically audible panel plus an external uncoded synthetic bm_lewis Nyx. Retain the source "
        "bank command, scene gain, music and room responses; the speech-only cue defines its own command clock. "
        "Use zero physical R, the actual demo event when playing, paired external-only and panel-plus-command "
        "controls on the identical noise. Count late and pre-external opens separately. Host native C161, no new exposure."))


if __name__=="__main__":
    main()
