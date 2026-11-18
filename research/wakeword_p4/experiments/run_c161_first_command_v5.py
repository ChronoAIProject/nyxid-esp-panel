"""Run the P4 FIFO and ESP Opus replay. Decode emitted packets and check the first command."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import correlate
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_coded_cue_p4_v5 import QEMU

WORK = Path(".local/wakeword/c161-first-command-v5")
BUILD = WORK / "build"
LOG = WORK / "first-command-uart.log"
STEM = "4090-c161-first-command-opus-v5"


def boot(build=BUILD, log=LOG, image=WORK / "first-command-flash.bin", marker=b"FIRST_COMMAND PASS"):
    images = read(build / "flasher_args.json")["flash_files"]
    flash = bytearray(b"\xff" * (4 << 20))
    for offset, name in images.items():
        start = int(offset, 16)
        data = (build / name).read_bytes()
        assert start + len(data) <= len(flash) and all(b == 255 for b in flash[start:start+len(data)])
        flash[start:start+len(data)] = data
    image.write_bytes(flash)
    log.unlink(missing_ok=True)
    cmd = [str(QEMU), "-machine", "esp32p4", "-display", "none", "-serial", f"file:{log.resolve()}",
           "-monitor", "stdio", "-snapshot", "-drive", f"file={image.resolve()},if=mtd,format=raw"]
    data = b""
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) as emulator:
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                data = log.read_bytes() if log.exists() else b""
                if marker+b"\r\n" in data or b"Rebooting..." in data or emulator.poll() is not None:
                    break
                time.sleep(.4)
            _,stderr = emulator.communicate(b"quit\n",timeout=8)
        finally:
            if emulator.poll() is None:
                emulator.kill()
    assert emulator.returncode == 0, stderr
    assert marker+b"\r\n" in data, data[-1800:]


def transcripts(reader, pcm):
    segments,_ = reader.transcribe(pcm.astype(np.float32)/32768,language="en",temperature=0,beam_size=3,
                                   vad_filter=False,condition_on_previous_text=False,word_timestamps=True)
    words = [dict(word=w.word,start=w.start,end=w.end) for s in segments for w in s.words or []]
    return dict(text="".join(w["word"] for w in words).strip(), words=words)


def decoder_api():
    lib = ctypes.CDLL("libopus.so.0")
    lib.opus_decoder_create.argtypes = [ctypes.c_int32, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.opus_decoder_create.restype = ctypes.c_void_p
    lib.opus_decode.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int32,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    lib.opus_get_version_string.restype = ctypes.c_char_p
    return lib


def decode(lib, packets):
    error = ctypes.c_int()
    decoder = lib.opus_decoder_create(16000,1,ctypes.byref(error))
    assert decoder and error.value == 0
    out = np.empty((len(packets),320),dtype="<i2")
    try:
        for i,p in enumerate(packets):
            assert lib.opus_decode(decoder,p,len(p),out[i].ctypes.data,320,0) == 320
    finally:
        lib.opus_decoder_destroy(decoder)
    return out.ravel()


def score(build=BUILD, log=LOG, stem=STEM, fixture_report=RESULTS / "4090-c161-coded-cue-v5-audit.json", fixture_dir=RESULTS, clocked=False):
    from faster_whisper import WhisperModel
    model = Path(".local/wakeword/whisper-small-en")
    plan = read(RESULTS / "4090-parallel-melo-context-asr-plan-v1.json")
    assert all(sha(p)==plan["files"][str(p)] for p in model.iterdir() if p.is_file())
    reader = WhisperModel(str(model),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    audit = read(fixture_report)
    witnesses = {r["channel"]:r for r in audit["witnesses"]}
    lib = decoder_api()
    rows = []
    current = None
    for line in log.read_text().splitlines():
        if line.startswith("BEGIN "):
            _,channel,delay = line.split()
            assert current is None
            current = (channel,int(delay))
            packets,starts = [],[]
            clock = None
        elif line.startswith("PACKET "):
            _,pts,source,packet = line.split()
            assert int(pts)==20*len(packets) and current is not None
            starts.append(int(source))
            packets.append(bytes.fromhex(packet))
        elif line.startswith("CLOCK "):
            _, channel, delay, time, step, prime = line.split()
            assert clocked and current == (channel, int(delay)) and clock is None
            clock = tuple(map(int, (time, step, prime)))
        elif line.startswith("END "):
            _,channel,delay,*metrics = line.split()
            assert current == (channel,int(delay))
            detected,handoff,first,stop,length,peak = map(int,metrics)
            wit = witnesses[channel]
            source_file = fixture_dir/wit["wav"]
            assert sha(source_file) == wit["sha256"]
            source,rate = sf.read(source_file,dtype="int16")
            start = wit["command_start"]
            assert rate==16000 and len(source)==length and detected==wit["notification"][0]["t"]
            if clocked:
                time, step, prime = clock
                assert (step, prime) == (wit["aec_chunk_samples"], wit["prime_samples"])
                assert (time + prime) % step == 0
                prior = ((time - step + prime) // 160) * 160 - prime
                assert handoff == ((time + prime) // 160) * 160 - prime and prior < detected <= handoff
            else:
                assert detected <= handoff < detected + 160
            assert first == handoff - 32000
            silence = int(delay)//20
            assert starts==[-1]*silence+list(range(first,stop,320)) and peak<64000
            full = decode(lib,packets)
            pcm = full[silence*320:]
            # Locate codec group delay using the source and actual received samples on the command.
            command = source[start:min(start+24000,length)].astype(np.float64)
            offset = start-first
            observed = pcm[offset:offset+len(command)+640].astype(np.float64)
            dots = correlate(observed,command,"valid",method="fft")
            squares = np.r_[0,np.cumsum(observed**2)]
            correlation = dots/np.sqrt(np.maximum(1,(squares[len(command):]-squares[:-len(command)])*(command@command)))
            lag = int(np.argmax(correlation))
            assert len(correlation)==641 and 0<lag<640
            decoded_cmd = pcm[offset+lag:]
            idx = (start-first)//320
            rows.append(dict(channel=channel,ready_delay_ms=int(delay),wav_sha256=wit["sha256"],detected=detected,
                  handoff=handoff,first_encoder_source_sample=first,command_onset=start,
                  command_packet_pts_ms=(idx+silence)*20,command_offset_in_packet=(start-first)%320,
                  encoder_source_end=stop,source_samples=length,peak_fifo_samples=peak,standin_silence_packets=silence,
                  total_packets=len(packets),decoded_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                  estimated_codec_delay_samples=lag,command_correlation=float(correlation[lag]),
                  command_source=transcripts(reader,source[start:]),command_decoded=transcripts(reader,decoded_cmd),
                  full_decoded=transcripts(reader,full)))
            if clocked:
                rows[-1]["capture_clock_at_handoff"] = time
            current = None
    assert current is None and {(r["channel"],r["ready_delay_ms"]) for r in rows}=={(c,d) for c in witnesses for d in (0,1500)}
    assert len(rows)==6 and "FIRST_COMMAND PASS" in log.read_text()
    archive = RESULTS/f"{stem}-uart.zip"
    with ZipFile(archive,"x",ZIP_DEFLATED) as packet_archive:
        packet_archive.write(log,"uart.log")
    component = WORK/"esp_audio_codec/lib/esp32p4/libesp_audio_codec.a"
    check = next(r for r in read(WORK/"codec-checksums.json")["files"] if r["path"]=="lib/esp32p4/libesp_audio_codec.a")
    assert sha(component)==check["hash"]
    save(RESULTS/f"{stem}.json",dict(fixture_report_sha256=sha(fixture_report),
        **({"frontend_report_sha256": audit["frontend_report_sha256"], "prepared_fixtures": audit["witnesses"]} if clocked else {}),
        fifo_source_sha256=sha(Path("components/panel_services/panel_voice_audio.c")),fixture_source_sha256=sha(RESULTS.parent/"first_command_p4/main/main.cpp"),
        qemu_sha256=sha(QEMU),elf_sha256=sha(build/"nyx_first_command.elf"),uart_sha256=sha(log),archive_sha256=sha(archive),
        codec_component_version="2.4.1",p4_codec_sha256=sha(component),codec_archive_sha256=sha(WORK/"codec.zip"),
        decoder=lib.opus_get_version_string().decode(),asr_files={p.name:sha(p) for p in model.iterdir() if p.is_file()},
        records=rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions",nargs="+",choices=("boot","score"))
    for action in parser.parse_args().actions:
        {"boot":boot,"score":score}[action]()
