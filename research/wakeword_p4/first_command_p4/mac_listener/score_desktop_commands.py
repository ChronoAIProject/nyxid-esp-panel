"""Check complete first-command delivery for the matched Nyx and Willow alert trials."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import struct

import numpy as np
import soundfile as sf

import check_bella_marker_only as stage
import check_desktop_alerts as trial

P = stage.P
OUT = trial.OUT.with_name("4090-c161-desktop-first-command.json")


def main():
    wake = P.read(trial.OUT)
    plan = P.read(trial.PLAN)
    assert P.sha(trial.PLAN)==wake["plan_sha256"]
    historical = P.read(stage.daily.OUT)
    codec = P.read(stage.daily.daily.OUT)
    assert P.sha(stage.clock.BUILD/"nyx_mac_listener.bin")==codec["capture_firmware_sha256"]
    assert P.sha(stage.prior.CODEC/"nyx_first_command.bin")==codec["codec"]["opus_sha256"]
    assert P.sha(stage.sender.BUILD/"board.bin")==historical["sources"]["board.bin"]
    assert P.sha(P.QEMU)==historical["qemu_sha256"]==wake["qemu_sha256"]
    processor, model, qid=stage.prior.load()
    assert qid==wake["qwen"] and qid["device"].endswith("RTX 4090")
    turbo=stage.prior.reader(wake["turbo"])
    decoder=stage.prior.decoder_api()
    rows=[]
    counts=defaultdict(Counter)
    with TemporaryDirectory(dir=trial.WORK) as temp:
        work=Path(temp)
        stage.clock.reference.WORK=work
        archive=OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for r,source in zip(wake["records"],plan["records"],strict=True):
                assert r["index"]==source["index"] and r["pcm_sha256"]==source["pcm_sha256"]
                name=r["target_id"]
                if name not in ("nyx","willow"):
                    continue
                i=r["index"]
                events=r["nyx"]["events"]["O"] if name=="nyx" else [(t+255)//256*256 for t in r["willow_native_samples"]]
                opened=events[0] if events else None
                row=dict(index=i,target=name,asset=r["asset"],command=r["command"],command_at=r["end"],
                         opened=opened,pcm_sha256=r["pcm_sha256"])
                count=counts[name+(".quiet" if r["asset"]=="clean" else ".alert")]
                count.update(cases=1,opened=int(opened is not None))
                if opened is not None:
                    mic,sr=sf.read(r["path"],dtype="int16")
                    assert sr==16000 and P.digest(mic)==r["pcm_sha256"]
                    label=f"{i:03}"
                    pcm,meta=stage.clock.capture(work,logs,i,label,mic,np.zeros_like(mic),opened)
                    cold,immediate,identity=stage.clock.opus(work,logs,i,label,pcm,decoder)
                    payload=struct.pack("<4I",i,opened,meta["first"],len(pcm))+pcm.tobytes()
                    log,(_, (sent,waited))=stage.sender.run_case(work,payload,cold)
                    logs.write(log,label+"/sender.log")
                    row.update(**meta,**identity,waited_sender=waited,sender_uart_sha256=P.sha(log),
                               sender_watchdog=log.read_text().count("Task watchdog got triggered"))
                    for mode,audio in (("immediate",immediate),("waited",stage.prior.decode(decoder,sent))):
                        a=stage.prior.evaluate(processor,model,audio,r["command"])
                        b=stage.prior.recognition(turbo,audio,r["command"])
                        row[mode]=dict(qwen=a,turbo=b)
                        count.update({mode+"_qwen":int(a["complete"]),mode+"_turbo":int(b["complete_suffix"])})
                rows.append(row)
                if len(rows)%8==0:
                    print("commands",len(rows),flush=True)
        assert len(rows)==2*(len(plan["assets"])+1)
        P.save(work/OUT.name,dict(records=rows,summary=counts,model=wake["model"],qwen=qid,turbo=wake["turbo"],
               capture_firmware_sha256=codec["capture_firmware_sha256"],codec=codec["codec"],
               sender_sha256=historical["sources"]["board.bin"],qemu_sha256=wake["qemu_sha256"],
               sources={p.name:P.sha(p) for p in (Path(__file__),trial.OUT,trial.PLAN,stage.daily.OUT,stage.daily.daily.OUT)},
               archive_sha256=P.sha(work/archive.name),policy="Only each listener's intended address, actual first "
               "open rounded to a 256-sample read. Use identical P4 call capture/Opus/send path for both. "
               "Complete 2 s history with silent R. Count all 27 assets per address, including missed calls. "
               "Immediate readiness and 500/1500 ms simulated connection/provider clocks."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(dict(counts),flush=True)


if __name__=="__main__":
    main()
