"""Check candidate event clocks across all previously paired daily interruptions."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import bella_marker_headroom as fit

P=fit.P
OUT=fit.OUT.with_name("4090-bella-headroom-history.json")


def main():
    receiver=fit.receiver
    previous=receiver.STEM.with_name(receiver.STEM.name+"-historical.json")
    prior=P.read(previous)
    old={r["label"]:r for r in prior["records"] if r["group"]=="external"}
    assert len(old)==192 and P.sha(previous.with_suffix(".zip"))==prior["archive_sha256"]
    fixed=receiver.test.fixed
    meta=P.read(receiver.WORK/"build.json")
    flash=bytearray((receiver.WORK/"guest.flash").read_bytes())
    assert fixed.digest(flash)==prior["build"]["flash_sha256"]==meta["flash_sha256"] and P.sha(fixed.QEMU)==meta["qemu_sha256"]
    cue,sr=sf.read(fit.WAV,dtype="int16")
    assert sr==16000 and P.sha(fit.WAV)==P.read(fit.OUT)["fit"]["cue_sha256"]
    playback=fit.early.playback
    original,_,_=playback.candidate()
    rows,files,counts=[],{},Counter()
    cache=fit.WORK/meta["flash_sha256"]
    with TemporaryDirectory(dir=fit.WORK) as folder, ZipFile(previous.with_suffix(".zip")) as saved:
        work=Path(folder)
        archive=OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for key,r,mic,ref in playback.playback.external(original,cue):
                label=key.replace("/plus2","/early")
                base=old[label]
                block=fixed.run_guest.payload(mic.tobytes(),ref.tobytes())
                code=fixed.digest(block)
                if r["role"]=="absent":
                    assert code==base["input_sha256"]
                    events=base["candidate"]["events"]
                    raw=saved.read("candidate/"+label+".log")
                    assert fixed.digest(raw)==prior["files"]["candidate/"+label+".log"]
                else:
                    epoch,log,faults=fixed.boot(work,flash,cache,block,1)
                    raw=log.read_bytes()
                    events=epoch[0]["events"]
                    for fault in faults:
                        name="retry/"+label+"/"+fault.name
                        files[name]=P.sha(fault)
                        logs.write(fault,name)
                assert b"BOARD PASS\r\n" in raw
                logs.writestr(label+".log",raw)
                files[label+".log"]=fixed.digest(raw)
                after=next(iter(events["O"]),None)
                before=next(iter(base["candidate"]["events"]["O"]),None)
                timely=after is not None and r["start"]<=after<r["command_at"]
                counts[r["role"]+"_cases"]+=1
                counts[r["role"]+"_opened"]+=int(after is not None)
                counts[r["role"]+"_timely"]+=int(timely)
                rows.append(dict(label=label,scene=r["scene"],start=r["start"],end=r["command_at"],original_open=before,
                                 events=events,input_sha256=code,clipped=r["clipped"], reused=r["role"]=="absent"))
        assert len(rows)==192
        P.save(work/OUT.name,dict(records=rows,files=files,counts=counts,archive_sha256=P.sha(work/archive.name),
                     cue_sha256=P.sha(fit.WAV),source_sha256=P.sha(previous),build=meta,script_sha256=P.sha(Path(__file__))))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
