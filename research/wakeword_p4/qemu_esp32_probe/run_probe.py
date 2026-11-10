"""Require actual official Hi ESP model creation and 100 zero-PCM detect calls.

This is runtime compatibility evidence, never recognition accuracy or timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('qemu', 'elf', 'nm', 'flash', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output directory')
    a.output.mkdir(parents=True)
    serial = (a.output/'serial.log').resolve()
    monitor = (a.output/'monitor.sock').resolve()
    # AF_UNIX paths have a short length limit; use the caller's short /tmp alias.
    if len(str(monitor).encode()) >= 104:
        p.error('Output path too long for monitor socket; use a short absolute path')
    symbols = subprocess.check_output([str(a.nm), str(a.elf)], text=True)
    address = re.search(r'^([0-9a-fA-F]+)\s+\w\s+nyx_wakenet_stage$', symbols, re.M)
    if not address:
        raise ValueError('Missing stage symbol')
    command = [str(a.qemu), '-machine', 'esp32', '-m', '4M', '-drive',
               f'file={a.flash.resolve()},if=mtd,format=raw', '-snapshot', '-nographic',
               '-serial', f'file:{serial}', '-monitor', f'unix:{monitor},server,nowait', '-no-reboot']
    started = time.monotonic()
    with (a.output/'host.log').open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                text = serial.read_text(errors='replace') if serial.exists() else ''
                if 'NYX_WN_ZERO_DONE' in text:
                    break
                if process.poll() is not None:
                    raise RuntimeError('QEMU exited before inference completion')
                if time.monotonic()-started > 60:
                    raise TimeoutError('Live QEMU did not finish compatibility probe in 60s')
                time.sleep(.1)
            with socket.socket(socket.AF_UNIX) as channel:
                channel.settimeout(3); channel.connect(str(monitor)); channel.recv(65536)
                channel.sendall(f'xp /1wx 0x{address[1]}\n'.encode())
                response = b''
                while b'(qemu)' not in response:
                    response += channel.recv(65536)
                (a.output/'stage-monitor.txt').write_bytes(response)
                if not re.search(rb':\s+0x00000068\b', response):
                    raise ValueError('Firmware stage is not104 (destroy/deinit completed)')
                channel.sendall(b'quit\n')
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                process.terminate(); process.wait(timeout=5)
    calls = re.findall(r'NYX_WN_ZERO_CHUNK i=(\d+) result=(\d+)', text)
    if calls != [(str(i), '0') for i in range(100)]:
        raise ValueError('Expected100 consecutive real detect calls with no zero-PCM activation')
    if 'NYX_WN_MODEL wn9_hiesp' not in text or 'chunk=512 channels=1 rate=16000' not in text:
        raise ValueError('Wrong model or audio interface')
    if 'Found 4MB PSRAM' not in text or 'NYX_QEMU_ONLY_RAM' in text:
        raise ValueError('Expected normal PSRAM initialization, no manual RAM shim')
    report = dict(role='Classic ESP32 official WakeNet9 Hi ESP runtime compatibility, not quality',
                  command=command, qemu_sha256=sha(a.qemu), elf_sha256=sha(a.elf),
                  flash_sha256=sha(a.flash), serial_sha256=sha(serial),
                  source_sha256=sha(Path(__file__)), firmware_stage=104, detect_calls=100,
                  model='wn9_hiesp', chunk=512, sample_rate=16000, channels=1,
                  threshold=.63, normal_emulated_psram=True, manual_ram_shim=False,
                  independent_final_audio_accessed=False,
                  limits='Zero PCM only. No positive recall, false-wake rate, real-device latency, P4 equivalence, or Hey Nyx parity established.')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
