"""Authenticate one locked development PCM for the official reference only."""
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


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('qemu','elf','nm','flash','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--record-index',type=int,required=True)
    p.add_argument('--model',choices=('willow','hiesp'),required=True)
    p.add_argument('--threshold',default='DEFAULT')
    p.add_argument('--timeout-seconds',type=float,default=0)
    a=p.parse_args()
    default=a.threshold=='DEFAULT'
    requested=0.0 if default else float(a.threshold)
    if not default and (not math.isfinite(requested) or not .4<=requested<=.9999):p.error('Invalid request')
    if a.output.exists():p.error('Fresh output required')
    plan_path=Path('research/wakeword_p4/results/4090-hey-official-development-execution-v1.json')
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(plan_path)])
    if remote!=plan_path.read_bytes():raise ValueError('Exact official plan must be pushed')
    plan=json.loads(remote)
    for name,path in plan['runtime'].items():
        if getattr(a,name)!=Path(path) or sha(Path(path))!=plan['files'][path]:
            raise ValueError('Unbound official runtime')
    if sha(Path(__file__))!=plan['files'][str(Path(__file__))]:raise ValueError('Unbound runner')
    if a.threshold not in plan['threshold_requests']:raise ValueError('Unregistered official threshold')
    if not 0<=a.record_index<len(plan['records']):raise ValueError('Wrong source index')
    row=plan['records'][a.record_index];wav=Path(row['path'])
    if sha(wav)!=row['sha256']:raise ValueError('Changed development source')
    with sf.SoundFile(wav) as f:
        if f.samplerate!=16000 or f.channels!=1 or f.subtype!='PCM_16':raise ValueError('Wrong PCM format')
        pcm=f.read(dtype='int16')
    raw=pcm.astype('<i2').tobytes();samples=len(pcm)
    if samples!=row['samples'] or hashlib.sha256(raw).hexdigest()!=row['pcm_sha256']:
        raise ValueError('Changed development PCM')
    a.output.mkdir(parents=True);output=a.output.resolve()
    monitor=output/'monitor.sock';serial=output/'serial.log'
    if len(str(monitor).encode())>=104:p.error('Use short output path')
    flags=int(default)+(2 if a.model=='willow' else 0)
    (output/'input.pcm').write_bytes(struct.pack('<IIfI',0x48595231,samples,requested,flags)+raw)
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
                # UART file writes can expose the marker before its counters.
                # Wait for the whole terminated line before quitting QEMU.
                if re.search(r'NYX_PCM_DONE samples=\d+ calls=\d+ events=\d+ verified=\d+ crc32=[0-9a-f]+\r?\n', text):
                    break
                if process.poll() is not None:
                    raise RuntimeError('QEMU exited before completion')
                if a.timeout_seconds > 0 and time.monotonic()-started > a.timeout_seconds:
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
    if (int(done[1]) != samples or int(done[2]) != (samples+511)//512 or
            int(done[3]) != len(events) or int(done[5],16) != zlib.crc32(raw)):
        raise ValueError('Firmware consumed bytes/counters differ from source')

    identity=re.search(r'HEY_REFERENCE abi=2 mode=(DEFAULT|SET) model=(\S+) request_bits=([0-9a-f]+) actual_bits=([0-9a-f]+) actual=([0-9.eE+-]+)',text)
    expected='wn9_heywillow_tts' if a.model=='willow' else 'wn9_hiesp'
    if not identity or identity[1]!=('DEFAULT' if default else 'SET') or identity[2]!=expected:
        raise ValueError('Wrong runtime/model/config identity')
    request_bits=struct.unpack('<I',struct.pack('<f',requested))[0]
    actual=struct.unpack('<f',struct.pack('<I',int(identity[4],16)))[0]
    if int(identity[3],16)!=request_bits or not math.isfinite(actual) or not 0<actual<1:
        raise ValueError('Invalid actual threshold bits')
    if struct.pack('<f',float(identity[5]))!=struct.pack('<f',actual):raise ValueError('Getter decimal/bits mismatch')
    prime=[(int(c),int(r)) for c,r in re.findall(r'HEY_REFERENCE_PRIME call=(\d+) result=(-?\d+)',text)]
    if [c for c,r in prime]!=list(range(1,64)) or any(r not in (-1,0,1) for c,r in prime):
        raise ValueError('Incomplete prime returns')
    after=re.search(r'HEY_REFERENCE_AFTER_PRIME actual_bits=([0-9a-f]+) actual=([0-9.eE+-]+)',text)
    if not after or after[1]!=identity[4] or struct.pack('<f',float(after[2]))!=struct.pack('<f',actual):
        raise ValueError('Missing or changed post-prime getter')
    returns=[(int(c),int(s),int(r)) for c,s,r in re.findall(r'HEY_REFERENCE_RETURN call=(\d+) sample_end=(\d+) result=(-?\d+)',text)]
    if len(returns)!=(samples+511)//512 or any(c!=i or end!=min(i*512,samples) or r not in (-1,0,1) for i,(c,end,r) in enumerate(returns,1)):
        raise ValueError('Incomplete ordered raw input returns')
    if sum(r==-1 for c,end,r in returns)!=int(done[4]):raise ValueError('Channel count differs')
    if [(c,end,r) for c,end,r in returns if r==1]!=[(e['call'],e['sample_end'],e['result']) for e in events]:
        raise ValueError('Raw returns differ from positive events')
    trace=output/'raw-returns.bin'
    trace.write_bytes(bytes(r+1 for c,end,r in returns))
    # Preserve every prime/input event; never turn a mismatch into a miss.
    if not re.search(r'NYX_PCM_PARAMETERS chunk=512 rate=16000 channels=1 set_result='+('0' if default else '1')+r' actual=',text):
        raise ValueError('Missing successful native API contract')
    report=dict(schema='hey-reference-qemu-development-v1',record_index=a.record_index,record=row,pcm_sha256=row['pcm_sha256'],model=expected,mode=identity[1],
                request_float32_bits=f'{request_bits:08x}',actual_float32_bits=identity[4],actual_threshold=actual,
                post_prime_getter_bits=after[1],raw_returns_path=trace.name,raw_returns_sha256=sha(trace),raw_returns_encoding="one byte per call, value=detect_return+1",prime_returns=prime,input_samples=samples,input_calls=int(done[2]),input_crc32=done[5],
                events=events,final_padding_samples=(-samples)%512,stage=104,
                qemu_sha256=sha(a.qemu),elf_sha256=sha(a.elf),flash_sha256=sha(a.flash),
                runner_sha256=sha(Path(__file__)),serial_sha256=sha(serial),
                execution_sha256=sha(plan_path),development_accessed=True,acceptance_accessed=False,limits='Official classic ESP32 QEMU development only; complete raw returns and getter/prime/CRC. No universal P4 equivalence or acceptance claim.')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
