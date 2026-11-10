"""Replay an explicitly marked developer WAV through unmodified Hi ESP detect."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import zlib
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'elf', 'nm', 'flash', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--record', required=True)
    p.add_argument('--threshold', type=float, default=.63)
    a = p.parse_args()
    if a.output.exists() or not math.isfinite(a.threshold) or not .4 <= a.threshold <= .9999:
        p.error('Use fresh output and supported threshold')
    manifest = json.loads(a.manifest.read_text())
    rows = [r for r in manifest['records'] if r['path'] == a.record]
    if len(rows) != 1 or rows[0]['role'] != 'development-transport-probe':
        raise ValueError('Require one explicit developer probe record')
    row = rows[0]; wav = (a.manifest.parent/row['path']).resolve()
    if not wav.is_relative_to(a.manifest.parent.resolve()) or sha(wav) != row['sha256']:
        raise ValueError('Escaped or changed audio')
    info = sf.info(wav); pcm, rate = sf.read(wav, dtype='int16'); raw = pcm.astype('<i2').tobytes()
    if rate != 16000 or pcm.ndim != 1 or info.subtype != 'PCM_16' or len(pcm) != row['samples']:
        raise ValueError('Need exact16k mono PCM16')
    if hashlib.sha256(raw).hexdigest() != row['pcm_sha256']:
        raise ValueError('Decoded PCM differs from manifest')
    a.output.mkdir(parents=True)
    output = a.output.resolve(); monitor = output/'monitor.sock'; serial = output/'serial.log'
    if len(str(monitor).encode()) >= 104:
        p.error('Use a shorter output path for AF_UNIX monitor')
    (output/'input.pcm').write_bytes(struct.pack('<IIfI', 0x4e595852, len(pcm), a.threshold, 0)+raw)
    symbols = subprocess.check_output([str(a.nm.resolve()), str(a.elf.resolve())], text=True)
    address = re.search(r'^([0-9a-fA-F]+)\s+\w\s+nyx_wakenet_stage$', symbols, re.M)
    if not address:
        raise ValueError('Missing completion symbol')
    command = [str(a.qemu.resolve()), '-machine', 'esp32', '-m', '4M', '-drive',
               f'file={a.flash.resolve()},if=mtd,format=raw', '-snapshot', '-nographic', '-semihosting',
               '-serial', f'file:{serial}', '-monitor', f'unix:{monitor},server,nowait', '-no-reboot']
    started = time.monotonic()
    with (output/'host.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                text = serial.read_text(errors='replace') if serial.exists() else ''
                if 'NYX_PCM_ERROR' in text or 'Guru Meditation' in text:
                    raise RuntimeError('Firmware failed; inspect preserved UART')
                if 'NYX_PCM_DONE' in text:
                    break
                if process.poll() is not None:
                    raise RuntimeError('QEMU exited before completion')
                if time.monotonic()-started > 120:
                    raise TimeoutError('Live QEMU exceeded developer probe deadline')
                time.sleep(.1)
            with socket.socket(socket.AF_UNIX) as channel:
                channel.settimeout(3); channel.connect(str(monitor))
                greeting = b''
                while b'(qemu)' not in greeting:
                    greeting += channel.recv(65536)
                channel.sendall(f'xp /1wx 0x{address[1]}\n'.encode())
                response = b''
                while b'(qemu)' not in response:
                    response += channel.recv(65536)
                (output/'stage-monitor.txt').write_bytes(response)
                if not re.search(rb':\s+0x00000068\b', response):
                    raise ValueError('Missing stage104')
                channel.sendall(b'quit\n')
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=5)
    done = re.search(r'NYX_PCM_DONE samples=(\d+) calls=(\d+) events=(\d+) verified=(\d+) crc32=([0-9a-f]+)', text)
    if not done:
        raise ValueError('Missing counters')
    events = [dict(call=int(c), sample_end=int(s), result=int(r), audio_seconds=int(s)/16000)
              for c,s,r in re.findall(r'NYX_PCM_EVENT call=(\d+) sample_end=(\d+) result=(\d+)', text)]
    if (int(done[1]) != len(pcm) or int(done[2]) != (len(pcm)+511)//512 or
            int(done[3]) != len(events) or int(done[5],16) != zlib.crc32(raw)):
        raise ValueError('Firmware consumed bytes/counters differ from source')
    parameters = re.search(r'NYX_PCM_PARAMETERS chunk=512 rate=16000 channels=1 set_result=1 actual=([0-9.]+)', text)
    if not parameters:
        raise ValueError('Missing successful threshold setter/getter evidence')
    report = dict(role='Hi ESP real-waveform developer transport probe, not acceptance',
                  record=row, manifest_sha256=sha(a.manifest), threshold_requested=a.threshold,
                  threshold_getter_after_set=float(parameters[1]),
                  model='wn9_hiesp', sample_rate=16000, chunk_samples=512,
                  input_samples=len(pcm), original_seconds=len(pcm)/16000,
                  final_chunk_zero_pad_samples=(-len(pcm)) % 512,
                  pcm_sha256=hashlib.sha256(raw).hexdigest(), pcm_crc32=f'{zlib.crc32(raw):08x}',
                  firmware_crc32_matched=True, firmware_stage=104, calls=int(done[2]), channel_verified_returns=int(done[4]), events=events,
                  timestamp_policy='End of consumed512-sample chunk, including explicit final zero padding; audio time, not host wall time.',
                  qemu_sha256=sha(a.qemu), elf_sha256=sha(a.elf), flash_sha256=sha(a.flash),
                  source_sha256=sha(Path(__file__)), serial_sha256=sha(serial), command=command,
                  independent_final_audio_accessed=False,
                  limits='Developer TTS training voices only. Text is not yet pronunciation audited. No representative recall/FAR, matched operating-point comparison, or hardware latency.')
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__':
    main()
