"""Check the fixed connection wait on all 96 paired daily interruptions at each target level."""
from collections import Counter
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import check_bella_daily_uplink as daily
import check_bella_sender as sender

P, prior = sender.P, sender.prior
WORK = Path(".local/wakeword/c161-bella-daily-sender")
OUT = P.RESULTS / "4090-c161-bella-daily-sender.json"


def measure(row, source, previous, logs, gpu):
    label = f"{row['index']:03}/{row['role']}"
    trace = sender.archived(source, label+"/capture.log", row["capture_uart_sha256"])
    index, opened, first, count = (row[k] for k in ("index","opened","first","samples"))
    assert P.values(trace,"OPEN") == [opened,opened//160*160,first]
    assert P.values(trace,"FINISH")[:3] == [row["stop"],first,count]
    pcm = prior.samples(trace,"PROCESSED",count)
    assert P.digest(pcm) == row["fifo_sha256"]
    trace = sender.archived(previous, label+"/opus.log", row["opus_uart_sha256"])
    cold = prior.packets(trace,index,count,0)
    payload = struct.pack("<4I",index,opened,first,count) + pcm.tobytes()
    log, ((original, _), (warm, waited)) = sender.run_case(WORK,payload,cold)
    assert original == cold and waited["captured"] == row["packets"]
    processor, model, turbo, decoder = gpu
    cold_audio = prior.decode(decoder,original)
    assert P.digest(cold_audio) == row["decoded_sha256"] == row["qwen"]["pcm_sha256"]
    received = prior.decode(decoder,warm)
    prefix = waited["sent_silence"]
    tail = received[prefix*320:]
    assert len(tail) == len(cold_audio)
    readings = {}
    for name, audio in (("waited",received),("waited_trimmed",tail)):
        readings[name] = dict(qwen=prior.evaluate(processor,model,audio,row["command"]),
                              turbo=prior.recognition(turbo,audio,row["command"]))
    logs.write(log, label+"/sender.log")
    return dict(**waited, fifo_sha256=row["fifo_sha256"],uart_sha256=P.sha(log),
                watchdog_warnings=log.read_text().count("Task watchdog got triggered"),
                changed_packets=sum(a!=b for a,b in zip(original,warm[prefix:],strict=True)),
                changed_decoded_samples=int(np.count_nonzero(cold_audio!=tail)),
                immediate=dict(qwen=row["qwen"],turbo=row["turbo"]),**readings)


def main():
    baseline, room = P.read(daily.OUT), P.read(sender.OUT)
    assert P.sha(sender.BUILD/"board.bin") == room["sources"]["board.bin"]
    assert P.sha(daily.TRACE) == baseline["capture_archive_sha256"]
    assert P.sha(daily.OUT.with_suffix(".zip")) == baseline["archive_sha256"]
    processor, model, qid = prior.load()
    assert qid == room["qwen"] == baseline["qwen"] and baseline["turbo"] == room["turbo"]
    decoder = prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == room["decoder"]
    assert P.sha(P.QEMU) == room["qemu_sha256"]
    gpu = processor, model, prior.reader(baseline["turbo"]), decoder
    rows, counts = [], {}
    WORK.mkdir(parents=True,exist_ok=True)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(daily.TRACE) as source, ZipFile(daily.OUT.with_suffix(".zip")) as previous:
        work = Path(tmp)
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for parent in baseline["records"]:
                row = {k:parent[k] for k in ("index","role","file","text","command","db","command_at","opened")}
                if row["opened"] is not None:
                    row["sender"] = measure(parent,source,previous,logs,gpu)
                rows.append(row)
                if len(rows)%24 == 0:
                    print(len(rows),"of 288",flush=True)
        assert len(rows) == 288
        for role in ("original","enhanced","willow"):
            cases = [r for r in rows if r["role"]==role]
            assert sorted(r["index"] for r in cases)==list(range(96))
            for condition in ("immediate","waited","waited_trimmed"):
                for reader, key in (("qwen","complete"),("turbo","complete_suffix")):
                    pairs = Counter((r["sender"]["immediate"][reader][key],r["sender"][condition][reader][key])
                                    for r in cases if r["opened"] is not None)
                    count = sum(n for (_,success),n in pairs.items() if success)
                    if condition == "immediate":
                        assert count == baseline["summary"][role+".all"][reader]
                    counts.setdefault(role,{}).setdefault(condition,{})[reader] = dict(
                        complete=count,opened=sum(pairs.values()),cases=len(cases),
                        gained=pairs[False,True],lost=pairs[True,False])
        with ZipFile(work/archive.name) as check:
            assert check.testzip() is None and len(check.namelist())==sum(r["opened"] is not None for r in rows)
        P.save(work/OUT.name,dict(records=rows,counts=counts,archive_sha256=P.sha(work/archive.name),
               qwen=qid,turbo=baseline["turbo"],decoder=room["decoder"],board_base=room["board_base"],
               model=room["model"],qemu_sha256=room["qemu_sha256"],sources={p.name:P.sha(p) for p in (
                   Path(__file__),Path(sender.__file__),sender.OUT,daily.OUT,sender.BUILD/"board.bin",
                   daily.TRACE,daily.OUT.with_suffix(".zip"))},
               policy="Retain all 96 consumed daily-overlap cases per role, including unopened. Use each actual "
                      "request and the original archived processed FIFO. Apply the 24-room sender protocol at "
                      "nominal 0/0 and 500/1500 ms connection/provider readiness from mic task entry. Verify "
                      "every immediate-ready packet against the matching archive. Read the complete sent stream "
                      "and its suffix after decoding the 50 leading stand-in frames. Simulated local peer."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(counts,flush=True)


if __name__ == "__main__":
    main()
