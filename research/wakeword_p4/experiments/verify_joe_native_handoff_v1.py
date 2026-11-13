"""Verify the original Joe tensors and every transported sample without inference."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import tarfile
import wave

import numpy as np
from scipy.signal import resample_poly


ARCHIVE_SHA = 'bbb934893d761e24df5a21da9ed59454739cc371204b8f237522c0bf164b8b24'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def committed(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'extracted', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.extracted.exists() and not args.output.exists()
    assert args.archive.stat().st_size == 30241016
    assert sha(args.archive.read_bytes()) == ARCHIVE_SHA
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == 372 and all(m.isfile() for m in members)
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
        manifest_blob = archive.extractfile('manifest.json').read()
        assert manifest_blob == committed('a7eaba7',
            'research/wakeword_p4/results/4090-joe-development-native-evidence-manifest-v1.json')
        entries = json.loads(manifest_blob)['files']
        assert len(entries) == len({e['path'] for e in entries}) == 371
        assert set(names) == {e['path'] for e in entries} | {'manifest.json'}
        for entry in entries:
            blob = archive.extractfile(entry['path']).read()
            assert len(blob) == entry['size'] and sha(blob) == entry['sha256']
        args.extracted.mkdir(parents=True)
        for member in members:
            path = args.extracted / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    root = args.extracted
    for path in (root / 'tools').iterdir():
        assert path.read_bytes() == committed('8ac36bf', 'research/wakeword_p4/experiments/' + path.name)
    original = root / 'joe-development-native-v1'
    transported = root / 'joe-development-native-transport-v1'
    assert (original / 'plan.json').read_bytes() == committed('9668ca7',
        'research/wakeword_p4/results/4090-joe-mike-generation-plan-v1.json')
    assert (original / 'generator.py').read_bytes() == (root / 'tools/generate_joe_development_native_v1.py').read_bytes()
    assert sha((original / 'records.jsonl').read_bytes()) == '91b310721d53642bea56fe838d57040785681ced5dce5b0b2f17b6c45a1a7e03'
    plan = json.loads((original / 'plan.json').read_text())
    model = next(m for m in plan['models'] if m['voice'] == 'joe')
    requests = [r for r in plan['requests'] if r['voice'] == 'joe']
    originals = [json.loads(line) for line in (original / 'records.jsonl').read_text().splitlines()]
    transports = [json.loads(line) for line in (transported / 'records.jsonl').read_text().splitlines()]
    assert len(requests) == len(originals) == len(transports) == 120
    details = []
    for request, source, row in zip(requests, originals, transports, strict=True):
        assert all(source[key] == row[key] == value for key, value in request.items())
        assert row['role'] == 'development' and row['voice'] == 'joe' and row['trimmed'] is False
        assert source['error'] == row['original_generator_error'] == 'AssertionError: '
        assert 'error' not in row
        assert all(row[key] == value for key, value in source.items() if key != 'error')
        assert source['model_sha256'] == model['expected_sha256']
        assert source['config_sha256'] == model['config_sha256']
        assert source['engine_tensor_path'] == request['id'] + '.engine.npy'
        assert row['raw_path'] == request['id'] + '.npy' and row['path'] == request['id'] + '.wav'
        for path, digest in ((original / source['engine_tensor_path'], source['engine_tensor_sha256']),
                             (transported / row['raw_path'], row['raw_sha256']),
                             (transported / row['path'], row['sha256'])):
            assert sha(path.read_bytes()) == digest
        tensor = np.load(original / source['engine_tensor_path'], allow_pickle=False)
        native = np.load(transported / row['raw_path'], allow_pickle=False)
        assert tensor.dtype == native.dtype == np.float32
        assert list(tensor.shape) == source['engine_output_shape'] == [1, 1, 1, len(native)]
        assert native.ndim == 1 and native.tobytes() == tensor.tobytes()
        assert np.isfinite(native).all() and np.max(np.abs(native)) < 1
        assert len(native) == row['native_samples'] and row['native_rate'] == 22050
        with wave.open(str(transported / row['path']), 'rb') as stream:
            assert (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000)
            blob = stream.readframes(stream.getnframes())
            pcm = np.frombuffer(blob, dtype='<i2')
            assert len(pcm) == stream.getnframes() == row['samples']
        divisor = math.gcd(row['native_rate'], 16000)
        full = resample_poly(native, 16000 // divisor, row['native_rate'] // divisor)
        assert np.isfinite(full).all() and np.max(np.abs(full)) < 1
        assert len(full) == len(pcm) == (len(native) * 16000 + row['native_rate'] - 1) // row['native_rate']
        assert row['generated_reference_end_sample'] == len(pcm)
        assert np.all((pcm != -32768) & (pcm != 32767)) and np.any(pcm)
        error = float(np.max(np.abs(pcm.astype(np.float64) / 32768 - full)))
        rms = float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768) ** 2)))
        assert error <= 1 / 32768 + 1e-7 and .2 <= len(pcm) / 16000 <= 5 and rms >= .001
        assert row['audit']['passed'] and all(row['audit']['checks'].values())
        details.append(dict(id=row['id'], native_samples=len(native), samples=len(pcm),
            pcm_sha256=sha(blob), maximum_transport_error=error, rms=rms))
    report = dict(schema='mac-joe-native-independent-receipt-v1',
        source_sha256=sha(Path(__file__).read_bytes()), archive_bytes=30241016,
        archive_sha256=ARCHIVE_SHA, manifest_sha256=sha(manifest_blob), payloads_verified=371,
        source_tools_and_plan_match_committed_bytes=True, original120_generator_failures_preserved=True,
        all120_ordered_requests_retained=True, all120_whole_tensor_sample_bytes_exact=True,
        all120_full_resampled_pcm_mechanical_checks_passed=True,
        target_counts=dict(Counter(r['target'] for r in transports)),
        duration_range_seconds=[min(d['samples'] for d in details) / 16000, max(d['samples'] for d in details) / 16000],
        maximum_transport_error=max(d['maximum_transport_error'] for d in details),
        versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy')}, records=details,
        wake_models_scored=False, audio_played_or_recorded=False, acceptance_claim=False,
        limits='Cross-machine byte and mechanical transport audit only. Standard-library wave reader, '
               'same rational polyphase resampling method; not an independent phonetic or quality judgment. '
               'No new synthesis, training, threshold selection, source-family independence or detector promotion.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('payloads_verified',
        'all120_full_resampled_pcm_mechanical_checks_passed', 'target_counts', 'maximum_transport_error')}))


if __name__ == '__main__':
    main()
