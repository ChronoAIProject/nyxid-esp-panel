"""Bind all 108 consumed clips to their unchanged, previously locked manifest."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import wave
import zlib

MANIFEST_SHA256 = '2294fe10ba8a7a72174083e4308a3f055b6c14f4f7dc4ba8b59807c229526bc0'
LITERALS = ('Hey Muse', 'Hey Willow', 'Hey Nicks')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pcm(path, row):
    assert not path.is_symlink() and sha(path.read_bytes()) == row['sha256']
    with wave.open(str(path), 'rb') as stream:
        assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) == (1, 2, 16000, 'NONE')
        assert stream.getnframes() == row['samples']
        data = stream.readframes(row['samples'])
        assert not stream.readframes(1)
    assert len(data) == 2 * row['samples'] and sha(data) == row['pcm_sha256']
    return data


def admit(manifest, wave_root):
    assert sha(manifest.read_bytes()) == MANIFEST_SHA256
    source = json.loads(manifest.read_text())
    assert source['schema'] == 'hey-development-replay-manifest-v1' and len(source['records']) == 360
    selected = [row for row in source['records'] if row['literal'] in LITERALS]
    assert len(selected) == 108 and Counter(row['literal'] for row in selected) == dict.fromkeys(LITERALS, 36)
    assert len({(row['source_id'], row['variant']) for row in selected}) == 108
    paths = list(wave_root.rglob('*.wav'))
    assert len(paths) == 108 and not any(path.is_symlink() for path in paths)
    by_hash = {sha(path.read_bytes()): path for path in paths}
    assert len(by_hash) == 108 and set(by_hash) == {row['sha256'] for row in selected}
    rows = []
    for row in selected:
        path = by_hash[row['sha256']]
        data = pcm(path, row)
        rows.append(dict(original=row, local_path=str(path.relative_to(wave_root)), pcm_crc32=zlib.crc32(data)))
    return dict(schema='muse-consumed108-fixtures-v1', source_manifest_sha256=MANIFEST_SHA256,
        source_records=360, rows=rows, selected_literals=list(LITERALS), source_labels_changed=False,
        evaluation_target_literal='Hey Muse', repeats=2, order='original selected manifest order, two adjacent fresh lifecycles per row',
        threshold='DEFAULT', setter_called=False, fresh_development_accessed=False, acceptance_accessed=False,
        limits='Only consumed one-voice diagnostic PCM; no new generation, threshold selection, FAR or candidate qualification.')


def load(report, root):
    assert report['schema'] == 'muse-consumed108-fixtures-v1' and report['source_manifest_sha256'] == MANIFEST_SHA256
    assert report['repeats'] == 2 and report['threshold'] == 'DEFAULT' and report['setter_called'] is False
    assert len(report['rows']) == 108
    payloads = []
    for entry in report['rows']:
        path = root / entry['local_path']
        assert path.resolve().is_relative_to(root.resolve())
        data = pcm(path, entry['original'])
        assert zlib.crc32(data) == entry['pcm_crc32']
        payloads.append(data)
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-manifest', 'wave-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = admit(args.source_manifest, args.wave_root)
    report['admission_source_sha256'] = sha(Path(__file__).read_bytes())
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(clips=len(report['rows']), exact_original_manifest=True, all_pcm_authenticated=True)))


if __name__ == '__main__':
    main()
