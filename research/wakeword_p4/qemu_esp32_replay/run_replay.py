"""Replay explicit development or preregistered final WAV through unmodified Hi ESP detect."""
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
    p.add_argument('--timeout-seconds', type=float, default=120.)
    p.add_argument('--final-preregistration', type=Path,
                   help='Required for explicitly frozen independent final PCM')
    a = p.parse_args()
    if (a.output.exists() or not math.isfinite(a.threshold) or not .4 <= a.threshold <= .9999
            or not math.isfinite(a.timeout_seconds) or a.timeout_seconds <= 0):
        p.error('Use fresh output and supported threshold')
    manifest = json.loads(a.manifest.read_text())
    rows = [r for r in manifest['records'] if r['path'] == a.record]
    if len(rows) != 1 or rows[0]['role'] not in ('development-transport-probe', 'development-negative-validation', 'development-read-speech-validation', 'development-paired-validation', 'independent-negative-final', 'independent-paired-final', 'independent-dialogue-final'):
        raise ValueError('Require one explicit supported replay record')
    row = rows[0]
    paired_final = row['role'] == 'independent-paired-final'
    dialogue_final = row['role'] == 'independent-dialogue-final'
    final = row['role'] == 'independent-negative-final' or paired_final or dialogue_final
    if bool(a.final_preregistration) != final:
        raise ValueError('Independent final replay requires a separate explicit preregistration')
    freeze = None
    if final:
        freeze = json.loads(a.final_preregistration.read_text())
        if (row.get('split') != 'testing'
                or sha(a.manifest) != freeze['pcm_manifest_sha256']
                or a.threshold != freeze['threshold_requested']
                or any(sha(getattr(a, k)) != freeze[k+'_sha256'] for k in ('qemu', 'elf', 'flash'))):
            raise ValueError('Final PCM, engine or operating point differs from preregistration')
        if dialogue_final:
            if (freeze.get('final_role') != 'independent-dialogue-final'
                    or not manifest.get('continuous_recordings') or manifest.get('inserted_intermediate_gaps')
                    or not manifest.get('original_pcm_preserved')
                    or manifest['source_audit_sha256'] != freeze['source_audit_sha256']
                    or manifest['source_manifest_sha256'] != freeze['source_manifest_sha256']
                    or row['frontend_final_zero_pad_samples'] != (-row['original_samples']) % 160
                    or row['samples'] != row['original_samples']+row['frontend_final_zero_pad_samples']
                    or row['source_seconds'] != row['original_samples']/16000
                    or not row['exact_original_pcm_preserved']):
                raise ValueError('Changed independent dialogue waveform protocol')
        elif paired_final:
            if (not manifest.get('paired_wakephrases')
                    or manifest['source_holdout_manifest_sha256'] != freeze['source_holdout_manifest_sha256']
                    or row['lead_samples'] != 28800 or row['tail_samples'] != 16000
                    or row['samples'] != row['source_samples']+44800):
                raise ValueError('Changed independent paired waveform protocol')
        elif (not manifest.get('all_frontend_frames_equal') or not manifest.get('utterance_concatenation')
              or manifest['frozen_negative_manifest_sha256'] != freeze['frozen_negative_manifest_sha256']
              or manifest['nyx_preregistration_sha256'] != freeze['nyx_preregistration_sha256']):
            raise ValueError('Changed independent negative waveform protocol')
    if row['role'] == 'development-paired-validation':
        if (row.get('split') != 'validation' or not manifest.get('paired_wakephrases')
                or not manifest.get('voice_partition_sha256') or not manifest.get('endpoint_annotation_source_sha256')
                or manifest.get('independent_final_audio_accessed')
                or not 0 < row.get('keyword_end', 0) <= row['samples']):
            raise ValueError('Need paired development validation with frozen endpoint metadata')
    if row['role'] in ('development-negative-validation', 'development-read-speech-validation'):
        source_policy = (manifest.get('continuous_recordings') if row['role'] == 'development-negative-validation'
                         else manifest.get('utterance_concatenation'))
        if (row.get('split') != 'validation' or not source_policy
                or not manifest.get('source_cache_sha256') or not manifest.get('source_audit_sha256')):
            raise ValueError('Need audited continuous development-validation negatives')
    wav = (a.manifest.parent/row['path']).resolve()
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
                if time.monotonic()-started > a.timeout_seconds:
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
    if final and float(parameters[1]) != freeze['threshold_getter_expected']:
        raise ValueError('Final engine getter differs from preregistration')
    report = dict(role=('Hi ESP independent paired fixed-point replay' if paired_final else 'Hi ESP independent negative fixed-point replay' if final else 'Hi ESP real-waveform developer transport probe, not acceptance'),
                  record=row, manifest_sha256=sha(a.manifest), threshold_requested=a.threshold,
                  threshold_getter_after_set=float(parameters[1]),
                  model='wn9_hiesp', sample_rate=16000, chunk_samples=512,
                  input_samples=len(pcm), original_seconds=len(pcm)/16000,
                  source_seconds=row.get('source_seconds', len(pcm)/16000),
                  development_negative_validation=row['role'] in ('development-negative-validation', 'development-read-speech-validation'),
                  utterance_concatenation=bool(manifest.get('utterance_concatenation')),
                  exposure_policy='Source seconds exclude inserted utterance gaps and alignment; all stream events counted.',
                  final_chunk_zero_pad_samples=(-len(pcm)) % 512,
                  pcm_sha256=hashlib.sha256(raw).hexdigest(), pcm_crc32=f'{zlib.crc32(raw):08x}',
                  firmware_crc32_matched=True, firmware_stage=104, calls=int(done[2]), channel_verified_returns=int(done[4]), events=events,
                  timestamp_policy='End of consumed512-sample chunk, including explicit final zero padding; audio time, not host wall time.',
                  qemu_sha256=sha(a.qemu), elf_sha256=sha(a.elf), flash_sha256=sha(a.flash),
                  source_sha256=sha(Path(__file__)), serial_sha256=sha(serial), command=command,
                  independent_final_audio_accessed=final,
                  final_preregistration_sha256=sha(a.final_preregistration) if final else None,
                  limits=('Independent dialogue negatives; repeated participants and transcript-screen limitations remain. Nominal Poisson bounds do not account for clustering. No positive recall, matched population FAR or hardware latency claim.' if dialogue_final else 'Independent paired TTS intention diagnostic; no validated phonetic endpoints or pronunciation labels. Not true recall or phonetic latency acceptance.' if paired_final else 'Independent negative audio only; fixed operating point, raw official events. No positive recall, matched population FAR or hardware latency claim.' if final else 'Development audio only. This records raw official detector events, with internal trigger policy. No independent acceptance, matched false-wake calibration or hardware latency.'))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report))


if __name__ == '__main__':
    main()
