"""Separate overlapping speech and music on the three weak calls restored by the louder cue."""
from collections import defaultdict
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import check_bella_daily_overlap as overlap
import check_bella_daily_sender as daily
import measure_command_capture as windows
import package_bella_daily_misses as weak

P, prior, sender = daily.P, daily.prior, daily.sender
clock = daily.daily.clock
WORK = Path(".local/wakeword/c161-bella-weak-interference")
OUT = P.RESULTS / "4090-c161-bella-weak-interference.json"


def scenes(work, parents):
    grid, back = overlap.rooms()
    rooms = {v[0]["file"]:v for cases in grid.values() for v in cases}
    lost = {r["index"] for r in parents if r["role"]=="original" and r["opened"] is None}
    assert len(lost)==3
    by_index = {r["index"]:r for r in parents if r["role"]=="enhanced"}
    with ZipFile(weak.PACK) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert manifest["source_sha256"] == P.sha(overlap.OUT)
        seen = set()
        for entry in manifest["records"]:
            if entry["role"]!="enhanced":
                continue
            i = int(entry["file"].split("/")[0])
            seen.add(i)
            row = by_index[i]
            case, s, render, ref, _, tracks = rooms[entry["source"]]
            assert row["file"]==case["file"] and row["db"]==entry["trial_db"]==-6
            assert case["end"]==s["command_at"]==row["command_at"]==entry["command_at"]
            assert [row["opened"]]==entry["simulated_open"]
            source = work/"speaker.wav"
            source.write_bytes(pack.read(entry["file"]))
            assert P.sha(source)==entry["sha256"]
            mixed = windows.channel(source,0)
            assert mixed.dtype == np.float64 and manifest["format"] == "PCM_24"
            mixed *= 2  # All three handoff WAVs use exactly half of the summed dry PCM.
            clean = tracks["enhanced"]
            start, stop = entry["speech_start"], entry["speech_end"]
            assert np.array_equal(clean[:start],mixed[:start]) and np.array_equal(clean[stop:],mixed[stop:])
            assert np.any(clean[start:stop]!=mixed[start:stop])
            command = clean[row["command_at"]:]
            for speech, dry in ((True,mixed),(False,clean)):
                external = overlap.board.bella.rear_filter(dry, back)
                for music in (True,False):
                    mic, clips = render(external,include_music=music)
                    if speech and music:
                        assert P.digest(mic)==entry["simulated_mic_sha256"]
                    yield row,entry,speech,music,mic,ref,clips,command
        assert seen==lost


def main():
    baseline, previous = P.read(daily.OUT), P.read(daily.daily.OUT)
    assert P.sha(daily.daily.OUT)==baseline["sources"][daily.daily.OUT.name]
    assert P.sha(clock.BUILD/"nyx_mac_listener.bin")==previous["capture_firmware_sha256"]
    assert P.sha(sender.BUILD/"board.bin")==baseline["sources"]["board.bin"]
    assert P.sha(prior.CODEC/"nyx_first_command.bin")==previous["codec"]["opus_sha256"]
    assert P.sha(P.QEMU)==baseline["qemu_sha256"]
    proc, model, qwen = prior.load()
    assert qwen==baseline["qwen"]
    turbo,decoder = prior.reader(baseline["turbo"]),prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==baseline["decoder"]
    WORK.mkdir(exist_ok=True,parents=True)
    clock.reference.WORK=WORK
    rows, counts = [],defaultdict(list)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for row,entry,speech,music,mic,ref,clips,command in scenes(work,baseline["records"]):
                i, opened = row["index"],row["opened"]
                condition = f"speech{int(speech)}_music{int(music)}"
                label = f"{i:03}/{condition}"
                pcm, info = clock.capture(work,logs,i,label,mic,ref,opened)
                cold, audio, codec = clock.opus(work,logs,i,label,pcm,decoder)
                payload = struct.pack("<4I",i,opened,info["first"],len(pcm))+pcm.tobytes()
                log, (_, (sent,waited)) = sender.run_case(WORK,payload,cold)
                logs.write(log,label+"/sender.log")
                received = prior.decode(decoder,sent)
                if speech and music:
                    assert P.digest(pcm)==row["sender"]["fifo_sha256"]
                    for name, x in (("immediate",audio),("waited",received)):
                        assert P.digest(x)==row["sender"][name]["qwen"]["pcm_sha256"]
                readings={}
                for name,x in (("immediate",audio),("waited",received)):
                    if speech and music:
                        reading = row["sender"][name]
                    else:
                        reading = dict(qwen=prior.evaluate(proc,model,x,row["command"]),
                                       turbo=prior.recognition(turbo,x,row["command"]))
                    readings[name]=dict(**reading,dry_windows=windows.locate(command,x))
                    counts[condition+"."+name].append(reading)
                rows.append(dict(index=i,condition=condition,file=row["file"],overlap_file=entry["file"],
                                 speech_end=entry["speech_end"],command_at=row["command_at"],command=row["command"],
                                 command_sha256=P.digest(command),mic_sha256=P.digest(mic),reference_sha256=P.digest(ref),
                                 clipped=clips,**info,opus_uart_sha256=codec["opus_uart_sha256"],
                                 opus_watchdog=codec["opus_watchdog"],sender_uart_sha256=P.sha(log),
                                 sender_watchdog=log.read_text().count("Task watchdog got triggered"),
                                 waited=waited,readings=readings))
                print(label,flush=True)
        assert len(rows)==len({(r["index"],r["condition"]) for r in rows})==12
        with ZipFile(work/archive.name) as check:
            assert check.testzip() is None and len([n for n in check.namelist() if not n.startswith("retry/")])==36
        summary={k:dict(cases=len(v),qwen=sum(x["qwen"]["complete"] for x in v),
                        turbo=sum(x["turbo"]["complete_suffix"] for x in v)) for k,v in counts.items()}
        files=(Path(__file__),Path(windows.__file__),weak.PACK,overlap.OUT,daily.OUT,daily.daily.OUT)
        P.save(work/OUT.name,dict(records=rows,summary=summary,qwen=qwen,turbo=baseline["turbo"],
               model=baseline["model"],board_base=baseline["board_base"],archive_sha256=P.sha(work/archive.name),
               capture_firmware_sha256=previous["capture_firmware_sha256"],codec=previous["codec"],
               sender_firmware_sha256=P.sha(sender.BUILD/"board.bin"),sources={p.name:P.sha(p) for p in files},
               policy="All three previously consumed weak enhanced calls recovered from the original-level misses. "
                      "Factorially mute the whole external interrupter or the music before microphone ADC. "
                      "Keep panel playback and its electrical reference, rear-room transfer, gain, enhanced address "
                      "and clean command samples, reference switching, and the full-mixture open fixed. "
                      "Decode the complete sent stream at both immediate and simulated 500/1500 ms readiness. "
                      "Baseline FIFO and both decoded streams reproduce the earlier hashes. "
                      "Score the known command suffix. Locate three waveform windows using the isolated dry "
                      "command, excluding the overlapping talk from the reference. Conditional on the fixed open."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(summary,flush=True)


if __name__=="__main__":
    main()
