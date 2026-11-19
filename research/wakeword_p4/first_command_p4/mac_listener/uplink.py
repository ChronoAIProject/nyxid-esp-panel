"""Send both matched first-command captures through the panel's P4 Opus encoder."""
from pathlib import Path
import re
import struct
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import replay as listener
from run_c161_first_command_v5 import BUILD, decode, decoder_api

STEM = listener.STEM + "-uplink"
BASE = listener.RESULTS/(listener.STEM+".json")


def main():
    before=listener.read(BASE)
    sources=listener.read(listener.rooms.PLAN)["records"]
    reader, wh=listener.load_reader()
    assert wh==before["asr"]
    lib=decoder_api()
    compressed=listener.RESULTS/(STEM+"-uart.zip")
    firmware=BUILD/"nyx_first_command.bin"
    directory=listener.WORK/("uplink-"+listener.sha(firmware)[:16])
    directory.mkdir(exist_ok=True)
    archived=listener.RESULTS/(listener.STEM+"-uart.zip")
    assert listener.sha(archived)==before["uart_sha256"]
    rows=[]
    with ZipFile(archived) as logs, ZipFile(compressed.with_suffix(".tmp"),"w",ZIP_DEFLATED) as out:
        for row in before["records"]:
            if row["first"]<0:
                continue
            i=row["index"]
            source=listener.load_wav(sources[i])
            raw=np.pad(source,(0,max(0,row["emitted"]-len(source))))[row["first"]:row["emitted"]]
            log=logs.read(f"{i:03}.log").decode().splitlines()
            processed=np.frombuffer(bytes.fromhex("".join(v.split()[2] for v in log if v.startswith("AUDIO "))),dtype="<i2")
            assert listener.digest(raw)==row["raw_sha256"] and listener.digest(processed)==row["processed_sha256"]
            length=row["count"]
            assert len(raw)==len(processed)==length
            name=f"{i:03}.log"
            uart=directory/name
            if not uart.exists() or "UPLINK PASS" not in uart.read_text():
                data=struct.pack("<3I",0x3552504f,i,length)+processed.tobytes()+raw.tobytes()
                listener.boot(BUILD,uart,directory/"flash.bin",b"UPLINK PASS",(2<<20,data))
            text=uart.read_text()
            record=dict(index=i, role=row["role"],snr_db=row["snr_db"],first=row["first"],count=length,
                        source_pcm_sha256=row["pcm_sha256"],uart_sha256=listener.sha(uart),lanes={})
            for lane,k in enumerate(("processed","raw")):
                groups=re.findall(rf"^UPLINK {i} {lane} {length}\n(.*?)^UPLINK_END {lane} (\d+)$",text,re.M|re.S)
                (section,n),=groups
                packets=[]
                for pts,start,b in re.findall(r"^PACKET (\d+) (\d+) ([0-9a-f]+)$",section,re.M):
                    assert (int(pts),int(start))==(len(packets)*20,len(packets)*320)
                    packets.append(bytes.fromhex(b))
                assert len(packets)==int(n)==(length+319)//320
                pcm=decode(lib,packets)
                actual=listener.transcripts(reader,pcm)
                words=listener.normalized(actual["text"])
                want=listener.normalized(row["carrier"])
                record["lanes"][k]=dict(packets=len(packets),bytes=sum(map(len,packets)),decoded_sha256=listener.digest(pcm),
                                        text=actual,exact_suffix=words[-len(want):]==want,
                                        before_codec=row["transcripts"][k]["exact_suffix"])
            rows.append(record)
            out.write(uart,name)
            print(i,len(rows),flush=True)
    assert len(rows)==sum(r["first"]>=0 for r in before["records"])
    compressed.with_suffix(".tmp").replace(compressed)
    listener.save(listener.RESULTS/(STEM+".json"),dict(records=rows,previous_sha256=listener.sha(BASE),asr=wh,
         firmware_sha256=listener.sha(firmware),code_sha256=listener.sha(Path(__file__)),
         firmware_source_sha256=listener.sha(listener.APP.parent/"main/main.cpp"),uart_sha256=listener.sha(compressed),
         decoder=lib.opus_get_version_string().decode(),policy="Original P4 FIFO capture versus raw mic at identical sample boundaries for every actual wake, two independent fresh P4 esp_audio_codec 2.4.1 Opus encoders. Mono 16 kHz, 20 ms, 24 kbit/s VBR, complexity 0, zero pad final packet only. Decode all packets in order from a new decoder; whole utterance ASR including codec delay. No packet loss; panel speaker silent."))


if __name__=="__main__":
    main()
