"""Evaluate the paired synthesis-context challenge without changing the trained gates."""
from collections import Counter, defaultdict
from pathlib import Path
import time
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import torch

import check_early_gate_voices as voices
import compare_synthesized_call_context as challenge
from score_paired_piper_willow import capture, host, uplink
from score_handoff_willow import willow_events

timing,ch = voices.timing,voices.ch
OUT = ch.RESULTS/(challenge.STEM+".json")


def main():
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    plan = ch.read(challenge.PLAN)
    plan_hash = ch.sha(challenge.PLAN)
    assert all(ch.sha(Path(path))==digest for path,digest in plan["files"].items())
    net = timing.spoken.natural.Detector()
    spec = ch.read(voices.gate.fit.COEFF)
    assert spec["models"]==net.identity
    costs = ch.read(timing.names.OUT)
    decoder = uplink.decoder_api()
    reader, asr = ch.load_reader()
    fw = uplink.firmware()
    rows,cases=[],[]
    totals=defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    tails={}
    archive=OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp,ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as logs:
        work=Path(tmp)
        uplink.WORK=work
        for s in plan["records"]:
            i=s["index"]
            assert i==len(rows)
            pcm=host.piper.load_wav(s)
            assert ch.digest(pcm)==s["pcm_sha256"]
            tail=ch.digest(pcm[s["end"]:])
            assert tail==s["suffix_sha256"]==tails.setdefault((s["voice"],s["carrier"]),tail)
            first=net.score(pcm)
            events,checks,jobs=voices.score(net,pcm,first["c161"],first["proposals"],first["events"],spec,costs)
            report=challenge.WORK/"runs"/f"{i:02}"/"report.json"
            deadline=time.monotonic()+180
            while not report.exists():
                if time.monotonic()>=deadline:
                    raise TimeoutError(report)
                time.sleep(1)
            willow=willow_events(i,s,plan_hash,logs,challenge.WORK/"runs")
            events["willow"]=[timing.block(t) for t in willow]
            scene=dict(index=i,voice=s["voice"],command=s["carrier"],role=s["target_id"],style=s["style"],
                       mic_sha256=s["pcm_sha256"],cue_at=s["start"],command_at=s["end"],willow=willow,
                       proposals=first["proposals"],c161=first["c161"],checks=checks,windows=jobs,routes={})
            cache={}
            for mode,ts in events.items():
                total=totals[s["target_id"]][s["style"]][mode]
                total["trials"]+=1
                if not ts:
                    scene["routes"][mode]=dict(events=[])
                    continue
                opened=ts[0]
                if opened not in cache:
                    cap_id=len(cases)
                    c=capture(reader,decoder,work,logs,cap_id,"context",pcm,opened,s["carrier"])
                    cases.append(dict(index=i,id=cap_id,opened=opened,**c))
                    cache[opened]=cap_id
                cap_id=cache[opened]
                complete=cases[cap_id]["lanes"]["selected"]["asr"]["complete_suffix"]
                timely=s["start"]<=opened<s["end"]
                total.update(hits=1,timely=timely,complete=complete,timely_complete=timely and complete)
                scene["routes"][mode]=dict(events=ts,opened=opened,lead=s["end"]-opened,capture=cap_id,timely=timely,complete=complete)
            rows.append(scene)
            if len(rows)%12==0:
                print(len(rows), len(cases),flush=True)
        retries=sorted(work.glob("*-attempt*.gz"))
        for path in retries:
            logs.write(path,"retry/"+path.name)
    assert len(rows)==108 and len(tails)==12
    with ZipFile(archive.with_suffix(".tmp")) as uart:
        assert uart.testzip() is None
        for c in cases:
            i=c["id"]
            text=uart.read(f"opus/{i:03}-context.log").decode().replace("\r\n","\n")
            for n,key in enumerate(("selected","raw")):
                packets=uplink.packets(text,i,c["emitted"]-c["first"],n)
                assert ch.digest(uplink.decode(decoder,packets))==c["lanes"][key]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT,dict(records=rows,captures=cases,summary=totals,asr=asr,models=net.identity,**fw,
            archive_sha256=ch.sha(archive),plan_sha256=plan_hash,coeff_sha256=ch.sha(voices.gate.fit.COEFF),
            score_sha256=ch.sha(Path(voices.__file__)),script_sha256=ch.sha(Path(__file__)),retries=[p.name for p in retries],
            policy="Prepublished PCM plan. Measure C161, frozen 360 ms and all three 180 ms gates and official "
            "Willow DEFAULT. Nyx verifier wall times estimated with preceding P4 P95. All routes delivered at "
            "the next 256-sample boundary, two-second FIFO, P4 Opus and unprompted small.en. Each source's "
            "independent opening encoded once; both returned streams decoded twice."))


if __name__=="__main__":
    main()
