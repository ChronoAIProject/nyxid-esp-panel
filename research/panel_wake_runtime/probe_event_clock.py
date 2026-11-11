"""Prepare and verify the silent full-app Hi ESP sample-clock probe."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import wave
import zlib

# Public paired Mac regression WAV. The pinned standalone UART WakeNet9 run
# (DET_MODE_95, 63 zero priming blocks) detected its first event at 46080.
WAV_SHA = '32dc4f97ca44785dfbaa8d37b516546e9c961660db4a09e7fa127b275379c114'
WAV_SAMPLES = 64286
WAV_CRC = 3365280488
PCM_SHA = 'e811775be216885f149166670eb79e29ec0570a1aff3952f29513a66aa718ab5'
READ_CHUNK = 256
DETECT_CHUNK = 512
EXPECTED_END = 46080
PREFIX = b'panel.wake.fixture: '
FIELDS = ('crc32', 'samples', 'read_chunk', 'detect_chunk', 'start_us', 'sample_end',
          'read_end', 'read_us', 'detect_start_us', 'detect_end_us')
RECORD = re.compile(b' '.join(k.encode() + rb'=(\d+)' for k in FIELDS))


def prepare(wav, output):
    if hashlib.sha256(wav.read_bytes()).hexdigest() != WAV_SHA:
        raise ValueError('Unexpected regression WAV hash')
    with wave.open(str(wav), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(),
                source.getnframes()) != (16000, 1, 2, WAV_SAMPLES):
            raise ValueError('Unexpected regression WAV format')
        pcm = source.readframes(WAV_SAMPLES)
    if len(pcm) != 2 * WAV_SAMPLES or zlib.crc32(pcm) != WAV_CRC:
        raise ValueError('Regression WAV does not match the standalone board run')
    # Only pad the tail, well after the reference event. Preserve every input
    # sample, including leading silence, and the original 16 kHz rate.
    pcm += bytes(2 * (-WAV_SAMPLES % DETECT_CHUNK))
    if hashlib.sha256(pcm).hexdigest() != PCM_SHA:
        raise ValueError('Unexpected probe PCM hash')
    with output.open('xb') as handle:
        handle.write(pcm)


def audit(pcm, uart, output):
    raw = pcm.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PCM_SHA:
        raise ValueError('Unexpected probe PCM or padding')
    log = uart.read_bytes()
    markers = [line.split(PREFIX, 1)[1].split(b'\x1b', 1)[0].strip()
               for line in log.splitlines() if PREFIX in line]
    match = RECORD.fullmatch(markers[0]) if len(markers) == 1 else None
    if not match:
        raise ValueError('Expected exactly one successful sample-clock probe record')
    row = dict(zip(FIELDS, map(int, match.groups())))
    if (row['crc32'], row['samples'], row['read_chunk'], row['detect_chunk'], row['sample_end'],
            row['read_end']) != (zlib.crc32(raw), len(raw) // 2, READ_CHUNK, DETECT_CHUNK,
                                 EXPECTED_END, EXPECTED_END):
        raise ValueError('Probe input, chunk geometry or trigger position differed from reference')
    if not 0 < row['start_us'] <= row['read_us'] <= row['detect_start_us'] <= row['detect_end_us']:
        raise ValueError('Probe timestamps out of order')
    result = {
        'scope': 'Synthetic PCM through the full app detector and sample-clock bookkeeping; no acoustic latency',
        'source_wav_sha256': WAV_SHA,
        'pcm_sha256': PCM_SHA,
        'standalone_reference_sample_end': EXPECTED_END,
        'on_target_probe': row,
        'detector_wall_us': row['detect_end_us'] - row['detect_start_us'],
    }
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(f'Full app and standalone detector agree: first event at sample {row["sample_end"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest='command')
    make = sub.add_parser('prepare')
    make.add_argument('wav', type=Path)
    make.add_argument('output', type=Path)
    check = sub.add_parser('audit')
    check.add_argument('pcm', type=Path)
    check.add_argument('uart', type=Path)
    check.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.wav, args.output)
    else:
        audit(args.pcm, args.uart, args.output)


if __name__ == '__main__':
    main()
