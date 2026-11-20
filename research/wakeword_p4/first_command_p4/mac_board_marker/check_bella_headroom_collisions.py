"""Attribute first wakes when two loudspeakers play the same released or candidate cue."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import bella_marker_headroom as fit

P, test = fit.P, fit.receiver.test
OUT = fit.OUT.with_name("4090-bella-headroom-same-cue-collisions.json")
PROTO = fit.study.ROOT / "experiments/bella_headroom_same_cue_collision_protocol.md"


def main():
    info=P.read(fit.OUT)
    test.WORK=fit.receiver.WORK
    fixed=test.fixed
    flash=bytearray((test.WORK/"guest.flash").read_bytes())
    assert fixed.digest(flash)==info["build"]["flash_sha256"] and P.sha(fixed.QEMU)==info["build"]["qemu_sha256"]
    parent=test.playback.playback
    controls=[r for r in P.read(parent.selfplay.OUT)["records"] if
              (r["reference"],r["version"])==("synchronous","original")]
    assert len(controls)==4
    cache=fit.WORK/info["build"]["flash_sha256"]
    rows,files,counts=[],{},defaultdict(Counter)
    with TemporaryDirectory(dir=fit.WORK) as folder:
        work=Path(folder)
        archive=OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for version,path,key in (("delivered",fit.early.WAV,"original_cue_sha256"),("candidate",fit.WAV,"cue_sha256")):
                cue,sr=sf.read(path,dtype="int16")
                assert sr==16000 and P.sha(path)==info["fit"][key]
                for r in controls:
                    geom=r["geometry"]
                    (ir,_,sha),_=parent.daily.tx.diag.room(geom)
                    assert sha==r["room_sha256"]
                    response=fftconvolve(cue.astype(float),ir)*r["scale"]
                    for offset in (-800,-400,0,400,800):
                        n=64000+len(response)+max(0,offset*16)
                        ext_start=32000+offset*16
                        panel=np.zeros(n)
                        ext=np.zeros(n)
                        panel[32000:32000+len(response)]=response
                        ext[ext_start:ext_start+len(response)]=response
                        total=np.rint(panel+ext)
                        mic=total.clip(-32768,32767).astype("<i2")
                        ref=np.zeros_like(mic)
                        ref[32000:32000+len(cue)]=cue
                        zero=np.zeros_like(ref)
                        alone=[]
                        for solo in (panel, ext):
                            single=np.rint(solo).clip(-32768,32767).astype("<i2")
                            alone.append(test.trace("candidate",single,zero).get("raw",[]))
                        trace=test.trace("candidate",mic,ref)
                        block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                        result,log,faults=fixed.boot(work,flash,cache,block,1)
                        events=result[0]["events"]
                        assert trace.get("allowed",[])==events["C"] and not events["M"]
                        first=next(iter(events["C"]),None)
                        if first is None:
                            cause="closed"
                        elif first in alone[0] and first in alone[1]:
                            cause="both_solo_clocks"
                        elif first in alone[1]:
                            cause="external_clock"
                        elif first in alone[0]:
                            cause="panel_clock"
                        else:
                            cause="mixture_only_clock"
                        label=f"{version}/{geom}/{offset}"
                        for p in (log,*faults):
                            k=label+("/wake.log" if p==log else "/"+p.name)
                            logs.write(p,k)
                            files[k]=P.sha(p)
                        counts[version+"/"+str(offset)][cause]+=1
                        rows.append(dict(label=label,version=version,geometry=geom,offset_ms=offset,
                                    panel_alone=alone[0],external_alone=alone[1],events=events,probe=trace,cause=cause,
                                    clipped=int(np.count_nonzero(total!=mic)),input_sha256=fixed.digest(block)))
        assert len(rows)==40
        P.save(work/OUT.name,dict(records=rows,counts=counts,files=files,build=info["build"],archive_sha256=P.sha(work/archive.name),
               sources={str(p):P.sha(p) for p in (Path(__file__),PROTO,parent.selfplay.OUT,fit.OUT)}))
        (work/OUT.name).replace(OUT)
        (work/archive.name).replace(archive)


if __name__=="__main__":
    main()
