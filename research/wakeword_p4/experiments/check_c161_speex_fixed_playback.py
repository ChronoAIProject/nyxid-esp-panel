"""Measure v5 self-play and simultaneous external playback on the floating-point P4 echo canceller."""
import io
import struct
from tempfile import TemporaryDirectory
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_lexical_echo import OUT
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v3_two_anchor_v1 import binding
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from check_c161_v5_playback_reference import AUDIT, BOT, RATE, ROOMS
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from replay_c161_pcm_cues_v1 import detect
from replay_c161_speex_p4 import parse
from run_c161_first_command_v5 import QEMU, boot, transcripts

STEM = "4090-c161-speex-fixed-playback"
START = RATE * 5
LENGTH = RATE * 10


def inputs():
    assert sha(WAV) == read(AUDIT)["cue_sha256"]
    cue, rate = sf.read(WAV,dtype="int16")
    assert rate == RATE
    rooms = read(ROOMS)
    commands = RESULTS/"4090-c161-v5-household-rooms-dry.zip"
    assert sha(commands) == rooms["waveform_archive_sha256"]
    with ZipFile(commands) as archive:
        self_cmd, r1 = sf.read(io.BytesIO(archive.read("bm_lewis-1-command.wav")),dtype="int16")
        ext_cmd, r2 = sf.read(io.BytesIO(archive.read("bm_lewis-0-command.wav")),dtype="int16")
    assert r1 == r2 == RATE
    bot = RESULTS/f"{BOT}-pcm-v1.npz"
    assert sha(bot)==read(RESULTS/f"{BOT}-v1.json")["pcm_archive_sha256"]
    with np.load(bot) as bank:
        plain = bank["bm_fable_digital"][:LENGTH].astype(float)
    onset = START + len(cue) + 320
    ref = plain.copy()
    self_clip = np.r_[cue,np.zeros(320),self_cmd]
    ref[START:START+len(self_clip)] = self_clip
    ext = np.zeros(LENGTH)
    ext_clip = np.r_[cue,np.zeros(320),ext_cmd]
    ext[START:START+len(ext_clip)] = ext_clip
    for scene in (0,1):
        room = next(r["room"] for r in rooms["records"] if r["scene"]==scene)
        path = Path(".local/wakeword/mit-rirs") / room
        assert sha(path)==rooms["files"][str(path)]
        h, rate=sf.read(path)
        assert rate==RATE
        h=h/np.linalg.norm(h)
        echo = .45*fftconvolve(ref,h)[:LENGTH]
        chat = .45*fftconvolve(plain,h)[:LENGTH]
        # A shared gain preserves every role's relative levels and the stereo clock.
        pairs={"self":(echo,ref),"external_over_reply":(ext+chat,plain),
               "simultaneous":(ext+echo,ref),"reply_only":(chat,plain)}
        gain=min(1.,28000/max(np.max(np.abs(v)) for pair in pairs.values() for v in pair))
        for role,(mic,reference) in pairs.items():
            mic,reference=(np.rint(gain*v).astype("<i2") for v in (mic,reference))
            yield f"{scene}-{role}",mic,reference,dict(role=role,room=room,gain=gain,echo_gain=.45,
                 start=START,command_start=onset,command_end=onset+len(ext_cmd),mic_sha256=digest(mic),ref_sha256=digest(reference))


def main():
    reader, identity = load_reader()
    build=OUT/"build-float"
    root=OUT/"fixed-cue"
    root.mkdir(exist_ok=True)
    archive=RESULTS/f"{STEM}-uart.zip"
    rows=[]
    with TemporaryDirectory() as tmp, ZipFile(root/"uart.zip","w",ZIP_DEFLATED) as uart:
        v5=binding(Path(tmp)/"v5.so",coded="v5")
        for key,mic,ref,meta in inputs():
            payload=struct.pack("<II",0x35434541,len(mic))+mic.tobytes()+ref.tobytes()
            log=root/f"{key}.log"
            boot(build,log,root/"flash.bin",b"SPEEX PASS",(2<<20,payload))
            clean,costs,heap=parse(log,len(mic))
            rowscore={}
            for stage,pcm in (("mic",mic),("reference",ref),("p4",clean)):
                hits=detect(v5,[pcm])
                row=dict(pcm_sha256=digest(pcm),events=hits)
                if hits and meta["role"] in ("external_over_reply","simultaneous") and stage!="reference":
                    start=max(0,(hits[0]["t"]+159)//160*160-32000)
                    text=transcripts(reader,pcm[start:meta["command_end"]])
                    expected=normalized("Turn off the desk lamp.")
                    row.update(capture_start=start,transcript=text,exact_command_suffix=normalized(text["text"])[-len(expected):]==expected)
                rowscore[stage]=row
            rows.append(dict(key=key,source=meta,stages=rowscore,internal=heap,cost_us=costs,uart_sha256=sha(log)))
            uart.write(log,log.name)
            print(key,[s["events"] for s in rowscore.values()],flush=True)
    assert len(rows)==8
    (root/"uart.zip").replace(archive)
    save(RESULTS/f"{STEM}.json",dict(cue_sha256=sha(WAV),audit_sha256=sha(AUDIT),rooms_sha256=sha(ROOMS),
         elf_sha256=sha(build/"nyx_speex_echo.elf"),qemu_sha256=sha(QEMU),uart_archive_sha256=sha(archive),asr=identity,records=rows))


if __name__=="__main__":
    main()
