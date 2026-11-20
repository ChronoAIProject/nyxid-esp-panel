"""Replay the everyday C161 conflict over the feature alignment and two silence lengths."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import check_bella_marker_only as cue
import check_desktop_alerts as theme

P = cue.P
OUT = P.RESULTS/"4090-c161-six-cups-clock.json"
WAV = P.RESULTS/"c161-six-cups-false-wake.wav"


def main():
    parent=P.read(theme.OUT)
    suspect, = [r for r in parent["records"] if r["target_id"]=="conversation" and r["nyx"]["events"]["O"]]
    mixed,sr=sf.read(suspect["path"],dtype="int16")
    assert sr==16000 and P.digest(mixed)==suspect["pcm_sha256"]
    expected=suspect["nyx"]["events"]
    assert expected["O"][0]<suspect["alert_at"]
    base=suspect["speech_at"]//480*480
    source={key:(r,pcm) for r,key,_,pcm in cue.overlap.event_inputs("daily") if r["text"]==suspect["text"]}
    assert len(source)==3 and suspect["speech_key"] in source and suspect["speech_at"]-base in range(0,480,80)
    assert P.sha(cue.GUEST)==parent["guest_sha256"] and P.sha(P.QEMU)==parent["qemu_sha256"]
    rows=[]
    work_root=theme.WORK/"six-cups"
    work_root.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=work_root) as tmp:
        work=Path(tmp)
        def cases():
            for key,(r,pcm) in source.items():
                for lead in (0,base):
                    for phase in range(0,480,80):
                        at=lead+phase
                        clean=np.zeros_like(mixed)
                        clean[at:at+len(pcm)]=pcm
                        replay=(key==suspect["speech_key"] and at==suspect["speech_at"])
                        if replay:
                            assert np.array_equal(clean[:suspect["alert_at"]],mixed[:suspect["alert_at"]])
                            sf.write(work/WAV.name,clean,16000,subtype="PCM_16")
                        label=f"{key}/{lead}/{phase}"
                        rows.append(dict(label=label,speech_key=key,voice=r["voice"],speech_pcm_sha256=P.digest(pcm),
                                         pcm_sha256=P.digest(clean),start=at,phase=phase,lead=lead,
                                         paired_alert_removed=replay))
                        yield label,clean.tobytes(),bytes(clean.nbytes),None,None
        guest=cue.overlap.board.run_guest.replay_cases(cue.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                      P.QEMU,work,cases())
        archive=OUT.with_suffix(".zip")
        counts=Counter()
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for row,r in zip(rows,guest["records"],strict=True):
                assert row["label"]==r["label"]
                row.update(r)
                counts[row["speech_key"]+"."+str(row["lead"])]+=bool(r["events"]["O"])
                if row["paired_alert_removed"]:
                    for route,ts in expected.items():
                        assert [t for t in r["events"][route] if t<=suspect["alert_at"]]==[t for t in ts if t<=suspect["alert_at"]]
                    audio,rate=sf.read(work/WAV.name,dtype="int16")
                    assert rate==16000 and P.digest(audio)==row["pcm_sha256"]
                log=work/"uart"/(r["input_sha256"]+".log")
                assert P.sha(log)==r["uart_sha256"]
                z.write(log,row["label"]+".log")
        P.save(work/OUT.name,dict(phrase=suspect["text"],alert=suspect["asset"],alert_at=suspect["alert_at"],
               alert_mixture=suspect["nyx"],model=parent["model"],guest_sha256=guest["guest_sha256"],
               qemu_sha256=guest["qemu_sha256"],records=rows,opened=counts,wav_sha256=P.sha(work/WAV.name),
               archive_sha256=P.sha(work/archive.name),sources={p.name:P.sha(p) for p in (Path(__file__),theme.OUT,cue.overlap.DAILY)},
               policy="All three existing same-voice renditions. Unaltered speech inserted after either zero or 52 "
                      "complete 480-sample blocks, plus one of six 80-sample offsets. Silent R, same post-input window. "
                      "Full 81,600-sample continuous warmup for each independent boot. The paired sample with the "
                      "alert removed is bit-identical to the alert mixture up to the scheduled alert start."))
        for dest in (archive,WAV,OUT):
            (work/dest.name).replace(dest)
    print(dict(counts),flush=True)


if __name__=="__main__":
    main()
