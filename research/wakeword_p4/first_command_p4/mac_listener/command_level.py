"""Replay all legacy far-room baseline/limited pairs via the P4 listener and Opus."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import replay as listener
from check_c161_command_playback_level import REFERENCE, RESULTS, STEM, digest, originals, recognition
from check_c161_v5_code_gain import binding
from run_c161_first_command_v5 import decode, decoder_api
from transition import CODEC, analyze, packets

BUILD = listener.WORK/"build-reference-prefix"


def playback_pack(records):
    archive = RESULTS/f"{STEM}-board.zip"
    manifest = []
    with ZipFile(archive, "x", ZIP_DEFLATED) as z:
        for meta, modes in originals():
            if (meta["role"],meta["channel"],meta["scene"]) != ("v5","digital",0) or meta["voice"] not in ("af_aoede","am_eric"):
                continue
            for name in ("baseline","limited"):
                # The external speaker plays only the dry source. Mac supplies room and microphone.
                start = meta["start"]
                track = modes[name][start:]
                mem = io.BytesIO()
                sf.write(mem,track,16000,format="WAV",subtype="PCM_16")
                data = mem.getvalue()
                wav = f"{meta['voice']}-lower-the-fan-speed-{name}.wav"
                z.writestr(ZipInfo(wav),data,compress_type=ZIP_DEFLATED)
                manifest.append(dict(wav=wav, voice=meta["voice"],samples=len(track), command_at=meta["end"]-start,
                    pcm_sha256=digest(track), wav_sha256=hashlib.sha256(data).hexdigest(), mode=name,
                    source_digital_pcm_sha256=digest(modes[name]), qemu_far_mic_sha256=next(
                        r["mic_sha256"] for r in records if r["voice"]==meta["voice"] and r["scene"]==0 and r["mode"]==name)))
        assert len(manifest)==4
        z.writestr(ZipInfo("manifest.json"),json.dumps(dict(sample_rate=16000,channels=1,format="PCM16",records=manifest),indent=2),
                   compress_type=ZIP_DEFLATED)
    with ZipFile(archive) as z:
        for row in manifest:
            data=z.read(row["wav"])
            pcm,rate=sf.read(io.BytesIO(data),dtype="int16")
            assert rate==16000 and len(pcm)==row["samples"] and digest(pcm)==row["pcm_sha256"]
            assert hashlib.sha256(data).hexdigest()==row["wav_sha256"]
    return listener.sha(archive)


def main(prior_logs=None):
    reader, identity = listener.load_reader()
    decoder = decoder_api()
    prior = {(r["source"]["voice"],r["source"]["scene"]):r for r in listener.read(REFERENCE)["records"]
             if r["source"]["channel"]=="far"}
    rows=[]
    saved = {}
    if prior_logs:
        with ZipFile(prior_logs) as prev:
            saved = {name:prev.read(name) for name in prev.namelist()}
    archive=RESULTS/f"{STEM}-p4-packets.zip"
    with TemporaryDirectory() as tmp, ZipFile(archive,"x",ZIP_DEFLATED) as z:
        work=Path(tmp)
        lib=binding(work/"cue.so",coded="v5")
        for meta, modes in originals():
            if meta["role"]!="v5" or meta["channel"]!="far":
                continue
            src=prior[meta["voice"],meta["scene"]]
            for mode in ("baseline","limited"):
                mic=modes[mode]
                i=len(rows)
                pair=np.column_stack((mic,np.zeros_like(mic)))
                payload=struct.pack("<3I",0x4452594e,i,len(mic))+pair.T.copy().tobytes()
                log=work/"listener.log"
                event_name=f"{i:02}-listener-events.log"
                opus_name=f"{i:02}-opus.log"
                if event_name in saved and opus_name in saved:
                    event_text=saved[event_name].decode()
                    opus_text=saved[opus_name].decode().replace("\r\n","\n")
                else:
                    retry_log = work/"failed.log"
                    try:
                        listener.boot(BUILD,log,work/"flash.bin",b"LISTENER PASS",(2<<20,payload))
                    except AssertionError:
                        if log.exists():
                            retry_log.write_bytes(log.read_bytes())
                            z.write(retry_log,f"{i:02}-listener-failed.log")
                        listener.boot(BUILD,log,work/"flash.bin",b"LISTENER PASS",(2<<20,payload))
                    text=log.read_text()
                    details,lanes=analyze(text,dict(group="far",index=i,command_at=meta["end"]),pair,lib)
                    tags=("READY ","HITS ","CUE_ROUTE ","OPEN ","C161_ALLOW ","V5_ALLOW ","FINISH ","LISTENER PASS")
                    event_text="\n".join(l for l in text.splitlines() if l.startswith(tags))+"\n"
                    n=details["count"]
                    enc=struct.pack("<3I",0x3552504f,i,n)+lanes["selected"].tobytes()+lanes["raw"].tobytes()
                    listener.boot(CODEC,log,work/"flash.bin",b"UPLINK PASS",(2<<20,enc))
                    opus_text=log.read_text()
                assert listener.values(event_text,"READY")==[i,len(mic),256,81600]
                opened,produced,first=listener.values(event_text,"OPEN")
                emitted,first_again,n,blocks,total,peak,over,heap=listener.values(event_text,"FINISH")
                expected=src["doubled"]["merged"][0]
                assert opened==(expected+255)//256*256 and first==first_again==max(0,produced-32000)
                assert n==emitted-first and first<meta["end"] and "UPLINK PASS" in opus_text
                z.writestr(event_name,event_text)
                z.writestr(opus_name,opus_text)
                r=dict(voice=meta["voice"],scene=meta["scene"],expected=src["expected"],mode=mode,
                       mic_sha256=digest(mic),command_at=meta["end"],delivered=opened,first=first,count=n,
                       blocks=blocks,peak_block_us=peak,over_budget=over,min_internal=heap,uplink={})
                for ix,label in enumerate(("selected","raw")):
                    pkts=packets(opus_text,i,n,ix)
                    pcm=decode(decoder,pkts)
                    r["uplink"][label]=dict(packets=len(pkts),decoded_sha256=digest(pcm),
                        asr=recognition(reader,pcm,src["expected"]))
                rows.append(r)
                print(i,mode,src["expected"],r["uplink"]["selected"]["asr"]["text"],flush=True)
    assert len(rows)==24
    listener.save(RESULTS/f"{STEM}-p4.json",dict(listener_sha256=listener.sha(BUILD/"nyx_mac_listener.bin"),
        opus_sha256=listener.sha(CODEC/"nyx_first_command.bin"),qemu_sha256=listener.sha(listener.QEMU),
        decoder=decoder.opus_get_version_string().decode(),asr_files=identity,
        packets_sha256=listener.sha(archive),board_archive_sha256=playback_pack(rows),records=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-logs",type=Path)
    main(parser.parse_args().prior_logs)
