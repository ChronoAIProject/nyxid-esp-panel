"""Validate private pilot captures and adapt them to the score-blind review tool."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import random
import re
import wave

import numpy as np
from scipy.signal import resample_poly

PROTOCOL = Path(__file__).with_name('protocol.json')
TEMPLATE_SHA = 'f3fa7017a128834dc385d2b2fe39c6431c94bf511beaef9b197f7b05c69720e0'
TARGET_RATE = 16000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def trials(spec):
    return [{'id': f'p{i}-{arm}', 'pair': i, 'arm': arm, 'text': spec['words'][arm] + pair['suffix']}
            for i, pair in enumerate(spec['pairs'], 1)
            for arm in (pair['first'], next(a for a in spec['words'] if a != pair['first']))]


def session(source, spec, fixture):
    data = json.loads(source)
    participant, condition = data['participant'], data['condition']
    if (data['protocol'] != spec['protocol'] or data['input_processing'] != 'disabled'
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,24}', participant)
            or condition not in spec['conditions'] or data['automated_browser'] is not fixture):
        raise ValueError('Unexpected capture provenance, condition or protocol')
    processed, geometry = [], set()
    for row, expected in zip(data['rows'], trials(spec), strict=True):
        if {k: row[k] for k in expected} != expected:
            raise ValueError('Changed, missing or reordered paired prompt')
        rate, channels = row['sample_rate'], row['input_channels']
        if type(rate) is not int or rate not in (16000, 32000, 44100, 48000, 88200, 96000):
            raise ValueError('Unsupported recording rate')
        if type(channels) is not int or channels not in (1, 2):
            raise ValueError('Unsupported microphone input geometry')
        geometry.add((rate, channels))
        pcm = base64.b64decode(row['pcm_s16le_base64'], validate=True)
        if len(pcm) != 2 * rate * spec['clip_seconds']:
            raise ValueError('Capture length does not match its sample rate')
        processed.append((expected, rate, channels, pcm))
    if len(geometry) != 1:
        raise ValueError('Capture changed its sample rate or channels mid-session')
    return participant, condition, processed


def pcm16k(pcm, rate):
    if rate == TARGET_RATE:
        return pcm
    ratio = math.gcd(rate, TARGET_RATE)
    source = np.frombuffer(pcm, dtype='<i2').astype(np.float64)
    out = resample_poly(source, TARGET_RATE // ratio, rate // ratio)
    if len(out) * rate != len(source) * TARGET_RATE:
        raise ValueError('Conversion changed clip duration')
    return np.clip(np.rint(out), -32768, 32767).astype('<i2').tobytes()


def write_wav(path, pcm, rate):
    with wave.open(str(path), 'wb') as wav:
        wav.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        wav.writeframes(pcm)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', action='append', type=Path, required=True)
    parser.add_argument('--review-template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test-fixture', action='store_true', help='Verify automation data; do not use for scoring')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh private directory')
    template = args.review_template.read_bytes()
    if sha(template) != TEMPLATE_SHA:
        raise ValueError('Unexpected endpoint review template')
    spec_bytes = PROTOCOL.read_bytes()
    spec = json.loads(spec_bytes)
    captures, manifest_rows, pack_rows, sessions = [], [], [], set()
    for file in args.session:
        blob = file.read_bytes()
        person, condition, rows = session(blob, spec, args.test_fixture)
        key = (person.casefold(), condition)
        if key in sessions:
            raise ValueError('Duplicate participant/condition; retain one complete session')
        sessions.add(key)
        captures.append((sha(blob), person, condition, rows))
    args.output.mkdir(parents=True, mode=0o700)
    audio = args.output / 'audio'
    audio.mkdir(mode=0o700)
    for capture_sha, person, condition, rows in captures:
        for expected, rate, channels, pcm in rows:
            identity = sha(f'{capture_sha}:{expected["id"]}'.encode())[:16]
            native = audio / f'{identity}-capture.wav'
            frozen = audio / f'{identity}.wav'
            canonical = pcm16k(pcm, rate)
            write_wav(native, pcm, rate)
            write_wav(frozen, canonical, TARGET_RATE)
            data = frozen.read_bytes()
            row = {'id': identity, 'participant': person, 'condition': condition, 'session_sha256': capture_sha,
                   'trial': expected, 'capture_rate': rate, 'input_channels': channels,
                   'capture_pcm_sha256': sha(pcm), 'capture_path': str(native.relative_to(args.output)),
                   'path': str(frozen.relative_to(args.output)), 'samples': len(canonical) // 2,
                   'wav_sha256': sha(data), 'pcm_sha256': sha(canonical)}
            manifest_rows.append(row)
            pack_rows.append({k: row[k] for k in ('id', 'samples', 'wav_sha256', 'pcm_sha256')} | {
                'rate': TARGET_RATE, 'intended_text': expected['text'],
                'audio': 'data:audio/wav;base64,' + base64.b64encode(data).decode('ascii')})
    if len({r['id'] for r in pack_rows}) != len(pack_rows):
        raise ValueError('Duplicate clip identity')
    manifest = args.output / 'manifest.json'
    write_json(manifest, {'role': 'automated-fixture' if args.test_fixture else 'human-pilot-capture-unreviewed',
                          'protocol_sha256': sha(spec_bytes), 'detector_outputs_read': False,
                          'canonical_rate': TARGET_RATE, 'records': manifest_rows})
    parent_sha = sha(manifest.read_bytes())
    random.Random(726).shuffle(pack_rows)
    pack = {'protocol': 'blind-human-endpoint-review-v1', 'parent_manifest_sha256': parent_sha,
            'rate': TARGET_RATE, 'records': pack_rows}
    pack['pack_id'] = sha(json.dumps(pack, separators=(',', ':')).encode())
    page = template.decode().replace('/* REVIEW_DATA */null', json.dumps(pack).replace('<', '\\u003c'))
    path = args.output / 'index.html'
    path.write_text(page)
    write_json(args.output / 'audit.json', {
        'role': 'automated-fixture' if args.test_fixture else 'human-pilot-blind-review-unreviewed',
        'template_sha256': TEMPLATE_SHA, 'html_sha256': sha(path.read_bytes()), 'pack_id': pack['pack_id'],
        'parent_manifest_sha256': parent_sha, 'clips': len(pack_rows),
        'mapping': [{'id': r['id'], 'path': r['path']} for r in manifest_rows],
        'detector_outputs_included': False, 'automatic_boundaries_included': False,
        'human_annotations_present': False})
    print(json.dumps({'sessions': len(captures), 'clips': len(pack_rows),
                      'duration_seconds_each': spec['clip_seconds'], 'automated_fixture': args.test_fixture}))


if __name__ == '__main__':
    main()
