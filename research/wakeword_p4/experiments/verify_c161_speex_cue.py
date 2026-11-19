"""Verify on-device v5 ticks against the earlier host matcher and package the exact stereo inputs."""
import hashlib
import io
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

from check_c161_speex_fixed_playback import OUT, RATE, STEM, inputs
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import boot, QEMU

PREFIX = STEM + "-device"


def main():
    baseline=RESULTS/f"{STEM}.json"
    previous={r["key"]:r for r in read(baseline)["records"]}
    root=OUT/"cue-p4"
    root.mkdir(exist_ok=True)
    build=OUT/"build-cue"
    rows=[]
    wav_path=RESULTS/f"{PREFIX}-stereo.zip"
    uart_path=RESULTS/f"{PREFIX}-uart.zip"
    with ZipFile(root/"uart.zip","w",ZIP_DEFLATED) as uart, ZipFile(root/"stereo.zip","w",ZIP_DEFLATED) as stereo:
        for key,mic,ref,meta in inputs():
            prior=previous.pop(key)
            assert prior["source"]==meta
            log=root/f"{key}.log"
            payload=struct.pack("<II",0x35434541,len(mic))+mic.tobytes()+ref.tobytes()
            boot(build,log,root/"flash.bin",b"SPEEX PASS",(2<<20,payload))
            pcm,costs,heap=parse(log,len(mic))
            hits=[int(n) for n in re.findall(r"SPEEX_CUE (\d+)\n",log.read_text())]
            reference=prior["stages"]["p4"]
            assert digest(pcm)==reference["pcm_sha256"] and hits==[e["t"] for e in reference["events"]]
            raw=io.BytesIO()
            wave=np.column_stack((mic,ref))
            sf.write(raw,wave,RATE,format="WAV",subtype="PCM_16")
            raw.seek(0)
            decoded,rate=sf.read(raw,dtype="int16")
            assert rate==RATE and np.array_equal(decoded,wave)
            data=raw.getvalue()
            stereo.writestr(key+".wav",data)
            uart.write(log,log.name)
            rows.append(dict(key=key,source=meta,stereo_wav_sha256=hashlib.sha256(data).hexdigest(),
                 uart_sha256=sha(log),output_pcm_sha256=digest(pcm),notifications=hits,cost_us=costs,internal=heap))
    assert not previous and len(rows)==8
    (root/"uart.zip").replace(uart_path)
    (root/"stereo.zip").replace(wav_path)
    save(RESULTS/f"{PREFIX}.json",dict(previous_sha256=sha(baseline),elf_sha256=sha(build/"nyx_speex_echo.elf"),
         qemu_sha256=sha(QEMU),detector_sha256=sha(RESULTS.parent/"portable/c161_coded_cue_v5.h"),
         uart_archive_sha256=sha(uart_path),stereo_archive_sha256=sha(wav_path),
         channels=["simulated microphone","synchronous loudspeaker reference"],sample_rate=RATE,records=rows))


if __name__=="__main__":
    main()
