"""Keep the foreground echo estimate fixed only while the local player emits the known v5 asset."""
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

from check_c161_speex_fixed_playback import AUDIT, OUT, RATE, START, STEM, inputs
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import boot, QEMU, transcripts

PREFIX = STEM + "-hold"


def main():
    reader, identity = load_reader()
    assert sha(WAV)==read(AUDIT)["cue_sha256"]
    cue,rate=sf.read(WAV,dtype="int16")
    assert rate==RATE
    source=RESULTS/f"{STEM}-device.json"
    prior={r["key"]:r for r in read(source)["records"]}
    root=OUT/"hold"
    root.mkdir(exist_ok=True)
    build=OUT/"build-hold"
    rows=[]
    archive=RESULTS/f"{PREFIX}-uart.zip"
    if archive.exists() or (RESULTS/f"{PREFIX}.json").exists():
        raise FileExistsError(PREFIX)
    with ZipFile(root/"uart.zip","w",ZIP_DEFLATED) as uart:
        for key,mic,ref,meta in inputs():
            original=prior.pop(key)
            assert meta==original["source"]
            played=np.rint(meta["gain"]*cue).astype("<i2")
            local=np.array_equal(ref[START:START+len(cue)],played)
            assert local==(meta["role"] in ("self","simultaneous"))
            # These are the local playback scheduler's bounds, available before rendering.
            begin,end=(START,START+len(cue)) if local else (0,0)
            payload=struct.pack("<II",0x35434541,len(mic))+mic.tobytes()+ref.tobytes()+struct.pack("<II",begin,end)
            log=root/f"{key}.log"
            boot(build,log,root/"flash.bin",b"SPEEX PASS",(2<<20,payload))
            pcm,costs,heap=parse(log,len(mic))
            text=log.read_text()
            assert f"SPEEX_LOCAL begin={begin} end={end}" in text
            hits=list(map(int,re.findall(r"SPEEX_CUE (\d+)\n",text)))
            if not local:
                assert digest(pcm)==original["output_pcm_sha256"] and hits==original["notifications"]
            row=dict(key=key,local_demo=[begin,end],notifications=hits,output_pcm_sha256=digest(pcm),
                     baseline_notifications=original["notifications"],cost_us=costs,internal=heap,uart_sha256=sha(log))
            if hits and meta["role"] in ("external_over_reply","simultaneous"):
                first=max(0,(hits[0]+159)//160*160-32000)
                words=transcripts(reader,pcm[first:meta["command_end"]])
                expected=normalized("Turn off the desk lamp.")
                row.update(first_sample=first,transcript=words,exact_command_suffix=normalized(words["text"])[-len(expected):]==expected)
            uart.write(log,log.name)
            rows.append(row)
            print(key,original["notifications"],hits,flush=True)
    assert not prior
    (root/"uart.zip").replace(archive)
    save(RESULTS/f"{PREFIX}.json",dict(controls_sha256=sha(source),elf_sha256=sha(build/"nyx_speex_echo.elf"),
         qemu_sha256=sha(QEMU),asr_files=identity,uart_archive_sha256=sha(archive),records=rows))


if __name__=="__main__":
    main()
