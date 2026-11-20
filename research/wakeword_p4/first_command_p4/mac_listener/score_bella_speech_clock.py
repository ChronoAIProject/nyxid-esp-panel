"""Read the complete first command on one daily interruption per matched playback file."""
from itertools import islice
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf

import check_bella_speech_clock as timing

cue, P = timing.cue, timing.P
OUT = P.RESULTS / "4090-c161-bella-speech-clock-uplink.json"


def main():
    parent = P.read(timing.OUT)
    wave, sr = sf.read(timing.WAV,dtype="int16")
    assert sr == 16000 and P.sha(timing.WAV) == parent["candidate_sha256"]
    assert P.digest(wave) == parent["dry"]["trim_advance"]["pcm_sha256"]
    baseline = P.read(cue.daily.OUT)
    original = {(r["index"],r["role"]):r for r in baseline["records"]}
    known = {r["index"]:r for r in parent["records"] if r["role"]=="daily"}
    assert set(known)==set(range(96))
    # One complete pass over the 24 playback files in generation order.
    selection = list(range(24))
    assert len({known[i]["file"] for i in selection})==24
    for db in (-6,0,6):
        assert sum(original[i,"original"]["db"]==db for i in selection)==8
    previous = P.read(cue.daily.daily.OUT)
    assert P.sha(cue.clock.BUILD/"nyx_mac_listener.bin")==previous["capture_firmware_sha256"]
    assert P.sha(cue.prior.CODEC/"nyx_first_command.bin")==previous["codec"]["opus_sha256"]
    assert P.sha(P.QEMU)==baseline["qemu_sha256"]==parent["qemu_sha256"]
    assert P.sha(cue.sender.BUILD/"board.bin")==baseline["sources"]["board.bin"]
    proc, model, qid = cue.prior.load()
    assert qid==parent["qwen"]==baseline["qwen"] and qid["device"].endswith("RTX 4090")
    turbo, decoder = cue.prior.reader(parent["turbo"]), cue.prior.decoder_api()
    rows=[]
    cand = "separated"
    versions = ("original","enhanced","willow",cand)
    cues = {v:[] for v in versions}
    cues[cand] = rows
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=timing.WORK) as tmp:
        work=Path(tmp)
        cue.clock.reference.WORK=work
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for i,(src,mic,ref,clips) in zip(selection,islice(cue.inputs(wave),len(selection)),strict=True):
                tested=known[i]
                assert i==src["index"] and P.digest(mic)==tested["mic_sha256"] and P.digest(ref)==tested["reference_sha256"]
                row=dict(index=i,file=src["file"],command=src["command"],db=src["db"],command_at=src["command_at"],
                         events=tested["events"],mic_sha256=P.digest(mic),reference_sha256=P.digest(ref),
                         clipped_samples=clips)
                for label in versions[:-1]:
                    cues[label].append(original[i,label].get("sender",{}))
                if row["events"]["O"]:
                    opened, = row["events"]["O"]
                    label = f"{i:03}"
                    pcm, meta = cue.clock.capture(work,logs,i,label,mic,ref,opened)
                    cold, immediate, ident = cue.clock.opus(work,logs,i,label,pcm,decoder)
                    payload = struct.pack("<4I",i,opened,meta["first"],len(pcm))+pcm.tobytes()
                    log, (_, (sent,waited)) = cue.sender.run_case(work,payload,cold)
                    logs.write(log,label+"/sender.log")
                    row.update(**meta,**ident,sender_uart_sha256=P.sha(log),waited_sender=waited,
                               sender_watchdog=log.read_text().count("Task watchdog got triggered"))
                    for mode, audio in (("immediate",immediate),("waited",cue.prior.decode(decoder,sent))):
                        row[mode] = dict(qwen=cue.prior.evaluate(proc,model,audio,src["command"]),
                                         turbo=cue.prior.recognition(turbo,audio,src["command"]))
                rows.append(row)
                print(i+1,"/24",flush=True)
        counts={}
        for version, entries in cues.items():
            assert len(entries)==len(selection)
            counts[version]={mode:{reader:sum(bool(row.get(mode,{}).get(reader,{}).get(key)) for row in entries)
                                   for reader,key in (("qwen","complete"),("turbo","complete_suffix"))}
                             for mode in ("immediate","waited")}
        with ZipFile(work/archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist())==len(set(logs.namelist()))
        P.save(work/OUT.name,dict(records=rows,counts=counts,selection=selection,sha256_wav=P.sha(timing.WAV),
               archive_sha256=P.sha(work/archive.name),qwen=qid,turbo=parent["turbo"],model=baseline["model"],
               capture_firmware_sha256=previous["capture_firmware_sha256"],codec=previous["codec"],
               sender_sha256=baseline["sources"]["board.bin"],qemu_sha256=parent["qemu_sha256"],
               sources={p.name:P.sha(p) for p in (Path(__file__),timing.OUT,cue.daily.OUT,cue.daily.daily.OUT)},
               policy="First complete rotation in source order over all 24 reused files. Eight per external level. "
                      "Keep each detector's own measured opening, include misses in the denominator. Use full P4 "
                      "capture and peer-bound encoded streams with immediate service or the simulated 500/1500 ms "
                      "connection/readiness times. Compare to the same 24 original, enhanced and Willow trials."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(counts,flush=True)


if __name__=="__main__":
    main()
