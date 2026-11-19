"""Exercise the pinned Mac C161 runtime and fixed v5 on the same raw mic and processed FIFO."""
import argparse
from pathlib import Path
import re
import struct
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP.parents[1]/"experiments"))
import c161_v5_household_rooms as rooms
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import boot, transcripts
from run_c161_coded_cue_p4_v5 import QEMU
from select_c161_synthesized_cue_v2 import load_wav

WORK = Path(".local/wakeword/c161-mac-listener")
STEM = "4090-c161-mac-listener-household"
MAC = "0d8e1935702a561b535a8c1a0921fd8034943d2c"
C161 = "c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81"
CUE = RESULTS/"hye-nyx-c161-synthetic-cue-v5.wav"


def values(text, name):
    fields, = re.findall(r"^"+name+r" ([0-9 -]+)$", text, re.M)
    return list(map(int,fields.split()))


def parse(log, i, src, pcm, reader):
    text=log.read_text()
    assert "LISTENER PASS" in text
    index,n,chunk,prime=values(text,"READY")
    assert index==i and n==len(pcm)==src["samples"] and prime==81600
    emitted,first,count,blocks,total_us,peak_us,over,heap=values(text,"FINISH")
    assert blocks==(n+chunk-1)//chunk and emitted==blocks*chunk//160*160
    events = [list(map(int,m.split())) for m in re.findall(r"^HITS ([0-9 -]+)$",text,re.M)]
    rows = dict(index=i, role=src["role"], scene=src["scene"], snr_db=src["command_snr_db"],
                voice=src["voice"],carrier=src["carrier"],pcm_sha256=src["pcm_sha256"],input_samples=n,
                prime=prime,chunk=chunk,events=events,first=first,emitted=emitted,count=count,
                mean_block_us=total_us/blocks,peak_block_us=peak_us,over_budget=over,blocks=blocks,min_internal=heap)
    audio = bytearray()
    for at,msg in re.findall(r"^AUDIO (\d+) ([0-9a-f]+)$",text,re.M):
        assert len(audio)==int(at)*2
        audio.extend(bytes.fromhex(msg))
    assert len(audio)==2*count
    if first<0:
        assert not events and not count and "OPEN " not in text
        return rows
    delivered,produced,opened=values(text,"OPEN")
    assert delivered==events[0][0] and produced==delivered//160*160 and opened==first==max(0,produced-32000)
    recorded=np.frombuffer(audio,dtype="<i2")
    assert count==emitted-first and first<n
    # Align lengths including the processing block's final zero padding.
    raw=np.pad(pcm,(0,max(0,emitted-n)))[first:emitted]
    assert len(raw)==len(recorded)
    command=normalized(src["carrier"])
    rows.update(delivered=delivered,command_at=src["command_at"],before_command=src["command_at"]-first,
                processed_sha256=digest(recorded),raw_sha256=digest(raw),transcripts={})
    for label,data in (("processed",recorded),("raw",raw)):
        reading=transcripts(reader,data)
        words=normalized(reading["text"])
        rows["transcripts"][label]=dict(**reading,exact_suffix=words[-len(command):]==command,
             contains_command=any(words[j:j+len(command)]==command for j in range(len(words))))
    return rows


def main(action):
    plan=read(rooms.PLAN)
    refs=RESULTS/f"{rooms.STEM}-v1.json"
    cue_ref=RESULTS/f"{rooms.STEM}-audio.json"
    baseline=read(refs)
    cues=read(cue_ref)
    assert baseline["plan_sha256"]==cues["plan_sha256"]==sha(rooms.PLAN)
    build=WORK/"build"
    firmware=build/"nyx_mac_listener.bin"
    model=WORK/"model/hey_nyx_model.bin"
    assert sha(model)==C161 and firmware.read_bytes().count(model.read_bytes())==1
    assert read(build/"config/sdkconfig.json")["NN_ANSI_C"]
    config=WORK/"model/manifest.json"
    assert read(config)["actual_prime_samples"]==81600
    elf_hash=sha(firmware)
    logs=WORK/elf_hash[:16]
    logs.mkdir(exist_ok=True)
    if action == "score":
        reader,identity=load_reader()
    records=[]
    for i,(src,prior,known) in enumerate(zip(plan["records"],baseline["records"],cues["records"],strict=True)):
        pcm=load_wav(src)
        assert i==known["index"] and digest(pcm)==src["pcm_sha256"]==prior["pcm_sha256"]
        log=logs/f"{i:03}.log"
        if action=="run":
            if not (log.exists() and "LISTENER PASS" in log.read_text()):
                payload=struct.pack("<3I",0x584e594d,i,len(pcm))+pcm.astype("<i2").tobytes()
                boot(build,log,WORK/"flash.bin",b"LISTENER PASS",(2<<20,payload))
        else:
            rec=parse(log,i,src,pcm,reader)
            actual=[r[1] for r in rec["events"] if r[1]>=0]
            rec.update(host_c161=prior["original_events"],host_v5=[e["t"] for e in known["events"]],
                       host_willow=prior["official_events"],log_sha256=sha(log))
            assert rec["host_v5"]==[r[2] for r in rec["events"] if r[2]>=0]
            rec["c161_identical"]=actual==rec["host_c161"]
            records.append(rec)
        if (i+1)%12==0:
            print(action,i+1,flush=True)
    if action=="score":
        archive=RESULTS/f"{STEM}-uart.zip"
        with ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED,compresslevel=3) as z:
            for i in range(len(records)):
                z.write(logs/f"{i:03}.log",f"{i:03}.log")
        archive.with_suffix(".tmp").replace(archive)
        runtime=WORK/"src/components/panel_wake_runtime"
        files=[APP/"main/listener.cpp",Path(__file__),runtime/"panel_wake_runtime.cpp",runtime/"panel_wake_runtime.h",
               model,config,CUE,WORK/"features.zip",WORK/"features/src/filterbank_util.c",rooms.PLAN,refs,cue_ref,
               Path("components/panel_services/panel_voice_audio.c"),APP.parents[1]/"portable/c161_coded_cue_v5.h"]
        save(RESULTS/f"{STEM}.json",dict(mac_commit=MAC,files={str(p):sha(p) for p in files},qemu_sha256=sha(QEMU),
             app_sha256=elf_hash,app_bytes=firmware.stat().st_size,asr=identity,uart_sha256=sha(archive),
             policy="One boot per existing input. Raw mic to Mac single C161 and frozen v5, zero idle playback reference to embedded ESP-SR AEC/AGC. First event wins at the end of an AEC read block. Transfer last 32000 adjusted samples into panel FIFO then append remaining processed audio. Device compares FIFO against its complete processed stream sample-by-sample. Continue running detectors for shadow comparisons. Speech-presence labels and transcripts do not enter device code. Final partial input read zero padded. QEMU reference FFT and esp-nn ANSI C.",records=records))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("run","score"))
    main(p.parse_args().action)
