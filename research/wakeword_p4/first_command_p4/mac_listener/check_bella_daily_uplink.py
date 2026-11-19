"""Decode the complete first command at each measured request during overlapping everyday speech."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import bella_daily_willow as official
import check_bella_daily_overlap as daily
import score_bella_room_willow as clock

P, prior = clock.P, clock.prior
OUT = P.RESULTS / "4090-c161-bella-daily-uplink.json"
WORK = Path(".local/wakeword/c161-bella-daily-uplink")
TRACE = WORK / "4090-c161-bella-daily-uplink-uart.zip"


def inputs():
    nyx = {(r["index"],r["role"]):r for r in P.read(daily.OUT)["records"]}
    for s,role,mic,ref,_ in daily.mixtures():
        r = nyx.pop((s["index"],role))
        assert P.digest(mic)==r["mic_sha256"] and P.digest(ref)==r["reference_sha256"]
        if role == "absent":
            assert not r["events"]["O"]
            continue
        opened = next(iter(r["events"]["O"]),None)
        yield r["index"],role,r,opened,mic,ref
    assert not nyx
    official_plan = P.read(official.PLAN)
    official_result = P.read(official.OUT)
    assert P.sha(official.PLAN) == official_result["plan_sha256"]
    grid,_ = daily.rooms()
    by_file = {value[0]["file"]:value for items in grid.values() for value in items}
    for s,r in zip(official_plan["records"],official_result["records"],strict=True):
        if s["target_id"] != "willow":
            continue
        _,meta,_,ref,_,_ = by_file[s["file"]]
        assert (s["index"],s["pcm_sha256"],s["end"],s["reference_sha256"]) == (
            r["index"],r["pcm_sha256"],meta["command_at"],P.digest(ref))
        if r["opened"] is None:
            mic = None
        else:
            mic, sr = sf.read(s["path"],dtype="int16")
            assert sr==16000 and P.digest(mic)==r["pcm_sha256"]
        yield s["index"],"willow",dict(s,command=meta["command"],command_at=s["end"],mic_sha256=s["pcm_sha256"]),r["opened"],mic,ref


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    clock.reference.WORK = WORK
    versions = P.read(clock.reference.OUT)
    assert P.sha(clock.BUILD/"nyx_mac_listener.bin") == P.read(clock.OUT)["capture_firmware_sha256"]
    assert P.sha(prior.CODEC/"nyx_first_command.bin") == versions["codec"]["opus_sha256"]
    assert P.sha(P.QEMU) == versions["codec"]["qemu_sha256"]
    processor, model, qid = prior.load()
    assert qid == versions["qwen"]
    turbo = prior.reader(versions["turbo"])
    decoder = prior.decoder_api()
    rows, counts = [],defaultdict(list)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/TRACE.name,"w",ZIP_DEFLATED) as capture, ZipFile(work/archive.name,"w",ZIP_DEFLATED) as codec:
            for i,role,s,opened,mic,ref in inputs():
                row = dict(index=i,role=role,file=s["file"],db=s["db"],text=s["text"],command=s["command"],
                           command_at=s["command_at"],opened=opened,mic_sha256=s["mic_sha256"],reference_sha256=s["reference_sha256"])
                if opened is not None:
                    label = f"{i:03}/{role}"
                    pcm, info = clock.capture(work,capture,i,label,mic,ref,opened)
                    _, audio, encoded = clock.opus(work,codec,i,label,pcm,decoder)
                    reading = prior.evaluate(processor,model,audio,s["command"])
                    assert reading["pcm_sha256"] == encoded["decoded_sha256"]
                    row.update(**info,**encoded,qwen=reading,turbo=prior.recognition(turbo,audio,s["command"]))
                rows.append(row)
                for sub in ("all",str(s["db"])):
                    counts[role+"."+sub].append(row)
                if len(rows)%24==0:
                    print(len(rows),"of 288",flush=True)
        summary = {}
        for name,group in counts.items():
            opened = [r for r in group if r["opened"] is not None]
            summary[name] = dict(cases=len(group),opened=len(opened),
                                 qwen=sum(r["qwen"]["complete"] for r in opened),
                                 turbo=sum(r["turbo"]["complete_suffix"] for r in opened))
        assert len(rows)==len({(r["index"],r["role"]) for r in rows})==288
        total = sum(r["opened"] is not None for r in rows)
        for name in (TRACE.name,archive.name):
            with ZipFile(work/name) as z:
                assert len([n for n in z.namelist() if not n.startswith("retry/")])==total and z.testzip() is None
        P.save(work/OUT.name,dict(records=rows,summary=summary,qwen=qid,turbo=versions["turbo"],codec=versions["codec"],
               capture_firmware_sha256=P.sha(clock.BUILD/"nyx_mac_listener.bin"),sources={p.name:P.sha(p) for p in (
                   daily.OUT,official.OUT,official.PLAN,clock.OUT,Path(daily.__file__),Path(__file__))},
               archive_sha256=P.sha(work/archive.name),capture_archive=TRACE.name,capture_archive_sha256=P.sha(work/TRACE.name),
               policy="Use the first measured board or official request for each target and all 96 consumed overlapping "
                      "speech cases in each denominator. Silent R through the open block, synchronous physical R thereafter, "
                      "full two-second processed history, 24 kb/s P4 Opus and both readers on complete decoded packets."))
        (work/archive.name).replace(archive)
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)
    print({k:v for k,v in summary.items() if k.endswith(".all")})


if __name__ == "__main__":
    main()
