"""Verify all frozen Joe presentations and full source/noise/RIR samples; no scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import wave

import numpy as np

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm(path):
    with wave.open(str(path), 'rb') as stream:
        assert (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) == (16000, 1, 2, 'NONE')
        blob = stream.readframes(stream.getnframes())
        data = np.frombuffer(blob, '<i2')
        assert len(data) == stream.getnframes()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'native', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert args.archive.stat().st_size == 28122937
    assert sha(args.archive) == '52a5ba59c6e2c5d38697cc14deba648b478c09d0e3d5f3d673d6b6842c3f8c58'
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == 368 and all(m.isfile() for m in members)
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
        manifest_blob = archive.extractfile('manifest.json').read()
        assert manifest_blob == subprocess.check_output(['git', 'show',
            'd5e6f61:research/wakeword_p4/results/4090-joe-development-render-evidence-manifest-v1.json'])
        entries = json.loads(manifest_blob)['files']
        assert len(entries) == len({e['path'] for e in entries}) == 367
        assert set(names) == {e['path'] for e in entries} | {'manifest.json'}
        for entry in entries:
            blob = archive.extractfile(entry['path']).read()
            assert len(blob) == entry['size'] and hashlib.sha256(blob).hexdigest() == entry['sha256']
        args.output.mkdir(parents=True)
        for member in members:
            path = args.output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    directory = args.output / 'joe-development-render-v1'
    manifest_path = directory / 'replay-manifest.json'
    assert sha(manifest_path) == 'edf6f7844f654d2a7e22f588873b917381aa8d0957b1e3309a60a0f1973895bc'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['records']) == 360
    assert manifest['scoring_authorized'] is False and manifest['independent_final_audio_accessed'] is False
    assert manifest['endpoint_reference'] == 'Complete generated waveform and full RIR tail, not human phonetic endpoints'
    records_blob = (directory / 'records.jsonl').read_bytes()
    assert hashlib.sha256(records_blob).hexdigest() == '87fe76043882146b29a1cd538b12f37378032826b97d479905ce504f78067887'
    rendered = [json.loads(line) for line in records_blob.decode().splitlines()]
    assert len(rendered) == 360
    for row, original in zip(manifest['records'], rendered, strict=True):
        assert row == {**{k: v for k, v in original.items() if k != 'audit'},
            'role': 'development-paired-validation', 'split': 'validation', 'voice': 'joe'}
        assert original['audit']['passed']
    assert sha(args.native / 'records.jsonl') == 'aadc0f099b29d0091c533e5642d9ca8ae28224a3807f4cab4d2139d26c6d5694'
    sources = {r['id']: r for r in map(json.loads, (args.native / 'records.jsonl').read_text().splitlines())}
    assert len(sources) == 120
    errors = []
    gains = []
    for row in manifest['records']:
        native = sources[row['source_id']]
        assert row['source_path'] == native['path'] == row['source_id'] + '.wav'
        assert sha(args.native / native['path']) == row['source_sha256'] == native['sha256']
        assert row['target'] == native['target'] and row['text'] == native['text']
        dry_pcm = pcm(args.native / native['path'])
        assert len(dry_pcm) == native['samples'] == row['source_samples']
        dry = dry_pcm.astype(np.float64) / 32768
        # A sparse, direct sum of all four delayed copies contrasts the FFT
        # renderer. Keep every sample, including the complete 960-sample tail.
        if row['variant'] == 2:
            speech = np.zeros(len(dry) + 960, np.float64)
            for delay, amplitude in ((0, 1.), (160, .25), (480, -.12), (960, .06)):
                speech[delay:delay + len(dry)] += amplitude * dry
        else:
            speech = dry
        expected = np.pad(speech, (28800, 16000))
        if row['variant']:
            seed = int(hashlib.sha256((row['source_id'] + ':joe-render-v1:' + str(row['variant'])).encode()).hexdigest()[:16], 16)
            assert seed == row['seed']
            noise = np.random.default_rng(seed).normal(0, 1, len(expected))
            snr = 25 if row['variant'] == 1 else 15
            noise *= np.sqrt(np.mean(dry ** 2)) * 10 ** (-snr / 20) / np.sqrt(np.mean(noise ** 2))
            expected += noise
            gain = min(1., .95 / max(float(np.max(np.abs(expected))), 1e-8))
        else:
            gain = 1.
        assert abs(gain - row['gain_after_mix']) <= 1e-14
        expected *= gain
        actual = pcm(directory / row['path'])
        assert sha(directory / row['path']) == row['sha256']
        assert hashlib.sha256(actual.tobytes()).hexdigest() == row['pcm_sha256']
        assert len(actual) == len(expected) == row['samples']
        assert row['native_generation_end'] == row['generated_output_end_sample'] == 28800 + len(dry)
        assert row['keyword_end'] == row['full_rir_end_sample'] == 28800 + len(speech)
        assert row['rir_tail_samples'] == (960 if row['variant'] == 2 else 0)
        assert row['samples'] == row['full_rir_end_sample'] + 16000
        assert np.isfinite(expected).all() and np.max(np.abs(expected)) < 1
        assert np.all((actual != -32768) & (actual != 32767))
        error = float(np.max(np.abs(actual.astype(np.float64) / 32768 - expected)))
        assert error <= 1 / 32768 + 1e-7
        errors.append(error)
        gains.append(gain)
    report = dict(schema='mac-joe-complete-render-admission-preparation-v1',
        source_sha256=sha(Path(__file__)),
        archive_bytes=args.archive.stat().st_size, archive_sha256=sha(args.archive), payloads_verified=367,
        manifest_sha256=sha(manifest_path), native_sources=120, complete_presentations=360,
        all_original_encoded_and_pcm_hashes_match=True, all_source_and_full_rir_noise_samples_reconstructed=True,
        maximum_quantization_error=max(errors), gain_range=[min(gains), max(gains)],
        numpy=np.__version__, qemu_invoked=False, wake_models_scored=False, accepted=False,
        limits='Complete frozen development transport and mechanical admission only. Sparse direct RIR '
               'reconstruction, same seeded noise method; no phonetic truth, recognition, actual engine '
               'identity, acoustic/hardware latency or independent acceptance.')
    (args.output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
