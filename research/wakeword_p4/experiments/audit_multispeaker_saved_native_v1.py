"""Independently verify all saved engine samples/PCM; never execute TTS."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

ROOT = Path('.local/wakeword/multispeaker-coverage-native-v1')
R = Path('research/wakeword_p4/results')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    summary_path = ROOT / 'summary.json'
    # Missing summary means not ready; never turn a partial log into a pass.
    summary = json.loads(summary_path.read_text())
    plan_path = R / '4090-multispeaker-coverage-native-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    assert sha(plan_path) == summary['plan_sha256'] == sha(ROOT / 'plan.json')
    assert summary['versions'] == plan['versions']
    assert sha(ROOT / 'generator.py') == plan['files'][str(Path(__file__).with_name('generate_multispeaker_coverage_native_v1.py'))]
    assert sha(ROOT / 'records.jsonl') == summary['records_sha256']
    rows = [json.loads(line) for line in (ROOT / 'records.jsonl').read_text().splitlines()]
    assert len(rows) == len(plan['requests']) == summary['requested'] == summary['recorded'] == 7740
    models = {m['voice']: m for m in plan['models']}
    for model in models.values():
        assert sha(model['model_path']) == model['sha256']
        assert sha(model['config_path']) == model['config_sha256']
    passed = 0
    files = 0
    failures = []
    maximum_error = 0.
    sample_counts = {a: 0 for a in ['narrow', 'broad']}
    for request, row in zip(plan['requests'], rows):
        assert all(row[k] == v for k, v in request.items())
        try:
            assert 'error' not in row
            model = models[request['voice']]
            assert row['model_sha256'] == model['sha256']
            assert row['config_sha256'] == model['config_sha256']
            config = json.loads(Path(model['config_path']).read_text())
            assert row['native_rate'] == config['audio']['sample_rate'] == 22050
            assert row['trimmed'] is False and row['random_seed'] is None
            assert row['phoneme_durations'] is None and row['duration_interface'] == plan['duration_interface']
            paths = []
            for field, digest in [('engine_tensor_path', 'engine_tensor_sha256'),
                                  ('raw_path', 'raw_sha256'), ('path', 'sha256')]:
                path = (ROOT / row[field]).resolve()
                assert path.is_relative_to(ROOT.resolve())
                assert sha(path) == row[digest]
                paths.append(path)
                files += 1
            tensor = np.load(paths[0], allow_pickle=False)
            native = np.load(paths[1], allow_pickle=False)
            assert tensor.dtype == native.dtype == np.float32
            assert list(tensor.shape) == row['engine_output_shape']
            assert tensor.ndim == 4 and tensor.shape[:3] == (1, 1, 1)
            assert native.ndim == 1 and native.size == row['native_samples'] > 0
            assert tensor.reshape(-1).tobytes() == native.tobytes()
            assert np.isfinite(native).all() and np.all(np.abs(native) < 1)
            divisor = math.gcd(16000, row['native_rate'])
            complete = resample_poly(native, 16000 // divisor, row['native_rate'] // divisor)
            pcm, rate = sf.read(paths[2], dtype='int16')
            info = sf.info(paths[2])
            assert rate == 16000 and info.subtype == 'PCM_16' and info.channels == 1
            assert pcm.ndim == 1 and np.any(pcm)
            assert len(pcm) == len(complete) == (native.size * 16000 + 22049) // 22050
            assert len(pcm) == row['samples'] == row['generated_reference_end_sample']
            assert .2 <= len(pcm) / 16000 <= 5
            assert np.isfinite(complete).all() and np.all(np.abs(complete) < 1)
            assert not np.any((pcm == -32768) | (pcm == 32767))
            decoded = pcm.astype(np.float64) / 32768
            assert np.sqrt(np.mean(decoded ** 2)) >= .001
            error = float(np.max(np.abs(decoded - complete.astype(np.float64))))
            assert error <= 1 / 32768 + 1e-7
            assert row['audit']['passed'] and all(row['audit']['checks'].values())
            maximum_error = max(maximum_error, error)
            for arm in request['arms']:
                sample_counts[arm] += len(pcm)
            passed += 1
        except Exception as exc:
            failures.append(dict(id=row['id'], error=f'{type(exc).__name__}: {exc}'))
    assert passed == summary['quality_passes']
    assert [f['id'] for f in failures] == summary['failures']
    assert summary['pack_qualified'] == (not failures)
    result = dict(schema='multispeaker-saved-native-independent-audit-v1',
                  auditor_sha256=sha(__file__), original_summary_sha256=sha(summary_path),
                  records_sha256=sha(ROOT / 'records.jsonl'), plan_sha256=sha(plan_path),
                  rows=len(rows), independently_qualified=passed, files_verified=files,
                  failures=failures, all_native_process_quality_pass=not failures,
                  maximum_pcm_serialization_error=maximum_error,
                  qualified_arm_transport_samples=sample_counts,
                  synthesis_calls=0, detector_invocations=0,
                  limits='Whole saved engine return and full PCM transport integrity only. Shared scipy resampling implementation; no stochastic regeneration, acoustic phoneme truth or physical/pretraining independence certification. Training-only data, not new holdout or candidate qualification.')
    with (R / '4090-multispeaker-saved-native-independent-audit-v1.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
