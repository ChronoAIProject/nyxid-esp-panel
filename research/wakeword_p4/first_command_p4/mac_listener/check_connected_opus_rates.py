"""Re-encode the same proven FIFO at 24/40 kb/s in the ESP codec on emulated P4."""
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import check_balanced_connected_commands as balanced

ch, mixed, tx = balanced.ch, balanced.mixed, balanced.tx
OUT = ch.RESULTS / "4090-c161-connected-opus-rates.json"
BUILD = Path(".local/wakeword/c161-opus-rate-comparison/build")


def inputs():
    complete = ch.read(balanced.OUT)
    older = ch.read(mixed.OUT)
    qwen_old = {r["index"]:r for r in ch.read(balanced.original.OUT)["records"]
                if r["variant"] == "marked_louder_command"}
    groups = mixed.study.sources()
    v5,_ = tx.study.panel_wave()
    quiet,rate = sf.read(mixed.panel.demo.WAVE,dtype="int16")
    bed,_,bh = tx.study.grid.room.rooms.read_noise("music")
    assert rate==16000 and ch.sha(mixed.panel.demo.WAVE)==complete["sources"][mixed.panel.demo.WAVE.name]
    tracks = (("balanced",balanced.PLAY,complete,"balanced_louder",balanced.OUT,"records"),
              ("original_weak",mixed.PLAY,older,"marked_louder_command",mixed.OUT,"guest"))
    for group,play,report,key,doc,lane in tracks:
        assert ch.sha(play)==report["speakers_sha256"] and ch.sha(doc.with_suffix(".zip"))==report["archive_sha256"]
        with ZipFile(play) as speakers, ZipFile(doc.with_suffix(".zip")) as logs:
            m = {r["index"]:r for r in json.loads(speakers.read("manifest.json"))["records"]}
            for row in report[lane]:
                s = row["source"]
                if group=="original_weak" and (row["variant"]!=key or s["external_db"]!=-6):
                    continue
                r = m[s["index"]]["tracks"][key]
                wave,sr = sf.read(BytesIO(speakers.read(r["file"])),dtype="int16")
                assert sr==16000 and ch.digest(wave)==r["pcm_sha256"]
                scene = mixed.room_scene(s,groups[s["voice"],s["command"]],quiet,v5,bed,bh)
                mic,_ = scene(wave)
                assert ch.digest(mic)==row["mic_sha256"]
                pcm = balanced.fifo_input(mic,row["finish"])
                cap = row["uplink"]
                if "fifo_sha256" in cap:
                    assert ch.digest(pcm)==cap["fifo_sha256"]
                before = row["qwen"]["complete"] if group=="balanced" else qwen_old[row["index"]]["qwen"]
                log = logs.read(f'{row["index"]:02}-opus.log').decode().replace("\r\n","\n")
                yield group,s,row,before,pcm,log


def main():
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    proc,model,qid = balanced.original.asr.load()
    previous = ch.read(balanced.OUT)
    assert qid == previous["qwen"] and uplink.firmware() == previous["codec"]
    turbo = tx.diag.reader(previous["turbo"])
    rows,stats = [],defaultdict(Counter)
    tmp = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work = Path(folder)
        uplink.WORK = work
        for group,s,row,before,pcm,old_log in inputs():
            i = len(rows)
            payload = struct.pack("<3I",0x3652504f,i,len(pcm))+pcm.tobytes()*2
            serial = work / "rate.log"
            uplink.boot(BUILD,serial,b"UPLINK PASS",payload,work)
            z.write(serial,f"{i:02}-opus.log")
            text = serial.read_text()
            original = uplink.packets(old_log,row["index"],len(pcm),0)
            baseline = uplink.packets(text,i,len(pcm),0)
            assert baseline == original
            candidate = uplink.packets(text,i,len(pcm),1)
            audio = uplink.decode(decoder,candidate)
            reading = balanced.evaluate(proc,model,audio,s["command"])
            whisper = uplink.recognition(turbo,audio,s["command"])
            counts = [sum(map(len,pkts)) for pkts in (baseline,candidate)]
            stats[f'{group}.{s["external_db"]}'].update(trials=1, matched_baseline_packets=len(baseline),
                bytes24=counts[0], bytes40=counts[1], qwen24=int(before), qwen40=int(reading["complete"]),
                qwen_gained=int(reading["complete"] and not before),qwen_lost=int(before and not reading["complete"]),
                turbo24=int(row["turbo"]),turbo40=int(whisper["complete_suffix"]),
                turbo_gained=int(whisper["complete_suffix"] and not row["turbo"]),
                turbo_lost=int(row["turbo"] and not whisper["complete_suffix"]))
            rows.append(dict(index=i, group=group, source=s, source_index=row["index"],opened=row["opened"],
                fifo_sha256=ch.digest(pcm),samples=len(pcm),packets=len(baseline), payload_bytes=counts,
                qwen24=before,turbo24=row["turbo"],qwen40=reading,turbo40=whisper))
            if len(rows)%12==0:
                print("Opus pairs",len(rows),flush=True)
        for f in sorted(work.glob("*-attempt*.gz")):
            z.write(f,"retry/"+f.name)
    assert len(rows)==48 and len(stats)==4
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    archive = OUT.with_suffix(".zip")
    tmp.replace(archive)
    src = Path(__file__).resolve().parent
    ch.save(OUT,dict(records=rows,summary=stats,qwen=qid,turbo=previous["turbo"],listener=previous["firmware"],
        codec_binary_sha256=ch.sha(BUILD/"nyx_first_command.bin"), original_codec=previous["codec"],
        archive_sha256=ch.sha(archive),sources={p.name:ch.sha(p) for p in (mixed.OUT,balanced.OUT,balanced.original.OUT,
        src/"build_opus_rate_comparison.sh",src.parent/"main/main.cpp",Path(__file__))},
        policy="Restore each selected FIFO from its original mic and capture clocks. Both emulated P4 encoder "
        "lanes receive identical samples with independent fresh state and the same 20 ms framing, complexity "
        "and VBR. Compare every 24 kb/s packet byte to its original received payload. Only lane 1 uses 40 kb/s. "
        "Payload bytes omit transport headers. Resource and network behavior require physical telemetry."))


if __name__ == "__main__":
    main()
