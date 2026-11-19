"""Replay repeated fixed cues and their aligned local reference on the original P4 listener."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import check_c161_pronunciation_ctc_v1 as ctc
import check_repeated_room_cue as repeat
from run_c161_first_command_v5 import transcripts

ch, grid = repeat.ch,repeat.grid
phase = repeat.drift.phase
OUT = ch.RESULTS/(repeat.source.STEM+"-p4.json")


def reading(dry,reader):
    info=ch.read(grid.fit.fixed.SOURCE_INFO)
    assert all(ch.sha(ctc.MODEL/k)==v["sha256"] for k,v in info["wav2vec2_files"].items())
    proc=Wav2Vec2Processor.from_pretrained(ctc.MODEL,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(ctc.MODEL,local_files_only=True).cuda().eval()
    out={}
    with torch.inference_mode():
        for name,pcm in dry.items():
            x=proc(pcm.astype("float32")/32768,sampling_rate=16000,return_tensors="pt")
            ids=model(input_values=x.input_values.cuda()).logits.argmax(-1)
            out[name]=dict(whisper=transcripts(reader,pcm)["text"],wav2vec2=proc.batch_decode(ids.cpu())[0])
    assert out["v5"]["whisper"]==info["readings"]["advance"]["whisper"]["text"] and out["v5"]["wav2vec2"]==info["readings"]["advance"]["wav2vec2"]
    return out


def main():
    corpus=ch.read(repeat.OUT)
    dry={"v5":grid.fit.original()[1],**repeat.tracks()}
    reader,asr=ch.load_reader()
    dry_read=reading(dry,reader)
    decoder=grid.gap.uplink.decoder_api()
    fw=phase.handoff.test.BUILD/"nyx_mac_listener.bin"
    codec=grid.gap.uplink.CODEC/"nyx_first_command.bin"
    history=ch.read(repeat.drift.STEM.with_suffix(".json"))
    assert (ch.sha(fw),ch.sha(codec))==(history["firmware_sha256"],history["codec_sha256"])
    stats=defaultdict(Counter)
    rows=[]
    pack=OUT.with_suffix(".zip")
    tmp=pack.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        work=Path(folder)
        full=ch.binding(work/"full.so",coded="v5")
        pair=ch.binding(work/"pair.so",coded="aligned")
        for (group,s,signals,_),o in zip(repeat.materials(),corpus["records"],strict=True):
            assert group==o["group"] and s==o["source"]
            # First, middle and last room in each independently named cohort at one difficult level.
            ids=grid.fit.EVAL if group=="initial" else repeat.source.EXTRA
            if s["room_id"] not in (ids[0],ids[len(ids)//2],ids[-1]) or (s["kind"],s["level_db"],s["player_ppm"])!=("music",-12,0):
                continue
            for name,x in signals.items():
                for local in (False,True):
                    j=len(rows)
                    ref=np.zeros_like(x)
                    if local:
                        ref[s["start"]:s["command_at"]]=grid.room.quantize(dry[name].astype(float)*10**(-12/20))
                    meta=dict(index=j,carrier=s["command"],cue_at=s["start"],command_at=s["command_at"])
                    expected=o["routes"][name]
                    for attempt in range(2):
                        try:
                            board=phase.probe(work,meta,x,ref,full,pair,j)
                            break
                        except (AssertionError,TimeoutError):
                            z.write(work/"serial.log",f"retry/{j:03}-{attempt}.log")
                            if attempt: raise
                    assert board["mic_sha256"]==expected["pcm_sha256"] and board["c161"]==expected["c161"]
                    assert board["fixed"]==[h["t"] for h in expected["fixed"]]
                    wanted=expected["self"]["opened"] if local else ([] if expected["opened"] is None else [expected["opened"]])
                    assert board["opened"]==wanted
                    z.writestr(f"events/{j:03}.txt","\n".join(board["uart_events"])+"\n")
                    stats[name]["self" if local else "external"]+=bool(wanted)
                    if wanted and not local:
                        audio=phase.handoff.bench.fifo(work,z,j,x,ref,board,reader,decoder,True,True)
                        board["uplink"]=audio
                        stats[name]["complete"]+=audio["recognition"]["complete_suffix"]
                        stats[name]["timely_complete"]+=audio["recognition"]["complete_suffix"] and wanted[0]<s["command_at"]
                    rows.append(dict(case=j,room=s["room"],kind=s["kind"],voice=s["voice"],command=s["command"],
                                    local=local,cue=name,result=board))
                    print(j,group,name,local,wanted,flush=True)
    assert len(rows)==36
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for r in rows:
            board=r["result"]
            if "uplink" in board:
                j,cap=r["case"],board["uplink"]
                uart=z.read(f"{j:02}-opus.log").decode().replace("\r\n","\n")
                count=cap["samples"]
                packets=grid.gap.uplink.packets(uart,j,count,0)
                assert ch.digest(grid.gap.uplink.decode(decoder,packets))==cap["pcm_sha256"]
    tmp.replace(pack)
    ch.save(OUT,dict(records=rows,summary=stats,dry_readings=dry_read,asr=asr,archive_sha256=ch.sha(pack),
        firmware_sha256=ch.sha(fw),codec_sha256=ch.sha(codec),qemu_sha256=history["qemu_sha256"],
        sources={p.name:ch.sha(p) for p in (repeat.OUT,repeat.source.PLAN,Path(__file__))},
        policy="Six deterministic room IDs, music, -12 dB cue and nominal source rate. Simulated P4 external "
        "and aligned self-play streams match host event clocks. Original mic and playback reference, "
        "P4 FIFO and Opus when externally opened. No hardware resource or speech generalization claim."))


if __name__ == "__main__":
    main()
