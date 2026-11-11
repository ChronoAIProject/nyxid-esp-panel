"""Replay frozen v30 on all public Mac regression sources, without audio I/O."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time
import zlib

import numpy as np
import soundfile as sf

from evaluate import Detector
from features import features
from detector_policy import smooth
from prepare_runtime_selftest import events

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'wakeword_baseline'))
from compare import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_result(scores, plan):
    window, threshold = plan['smoothing_steps'], plan['v30_threshold']
    host = smooth(scores, window)
    # Integer/256 scores have exact double sums; divide once as on the P4.
    device = np.convolve(scores.astype(np.float64), np.ones(window), mode='valid')/window
    steps = [events(values, threshold, window) for values in (host, device)]
    result = []
    for detected in steps:
        offsets = [plan['v30_block_samples']*(step+1)-plan['v30_prime_samples'] for step in detected]
        source = [at for at in offsets if at > 0]
        result.append({'events': len(source), 'offsets': source,
                       'one_based_score_steps': detected,
                       'prime_event_offsets': [at for at in offsets if at <= 0]})
    return result


def groups(rows, target):
    return {'all': summarize(rows, target), 'by_voice_and_variant': {
        f'{voice}/variant-{variant}': summarize(
            [row for row in rows if row['voice'] == voice and row['variant'] == variant], target)
        for voice, variant in sorted({(row['voice'], row['variant']) for row in rows})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'model-dir', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    profile_path = Path(__file__).resolve().parents[2]/'panel_wake_runtime/plans/gpu-v30-runtime-v1.json'
    assert sha(profile_path) == plan['profile_sha256']
    profile = json.loads(profile_path.read_bytes())
    model = args.model_dir/'hey_nyx.tflite'
    calibration = args.model_dir/'4090-v30-expanded-calibration.json'
    assert sha(model) == profile['model_sha256']
    assert sha(calibration) == profile['calibration_sha256']
    assert sha(args.model_dir/'training.json') == profile['training_report_sha256']
    for key, expected in plan['host_versions'].items():
        assert version(key) == expected, key
    manifest, baseline = args.sources/'manifest.jsonl', args.sources/'baseline.json'
    assert sha(manifest) == plan['manifest_sha256'] and sha(baseline) == plan['baseline_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    official = json.loads(baseline.read_bytes())
    assert len(rows) == len(official['clips']) == plan['source_count']
    assert dict(Counter(row['target'] for row in rows)) == plan['target_counts']
    assert sorted({row['voice'] for row in rows}) == plan['voices']
    assert sorted({row['variant'] for row in rows}) == plan['variants']
    assert official['manifest_sha256'] == plan['manifest_sha256'] and official['restored']
    assert (profile['threshold'], profile['smoothing_steps'], profile['refractory_steps']) == (
        plan['v30_threshold'], plan['smoothing_steps'], plan['refractory_steps'])
    prime = (profile['input_frames']+200)*160
    assert prime+(-prime)%480 == plan['v30_prime_samples']
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.plan, args.output/'plan.json')
    shutil.copyfile(__file__, args.output/'source.py')
    arrays, detailed, host_rows, device_rows = {}, [], [], []
    detector = Detector(model)
    started = time.monotonic()
    for index, (row, measured) in enumerate(zip(rows, official['clips'], strict=True)):
        assert all(measured[k] == v for k, v in row.items())
        assert measured['result']['complete']
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = args.sources/relative
        assert sha(path) == row['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == plan['sample_rate'] and pcm.ndim == 1 and len(pcm) == row['samples']
        assert zlib.crc32(pcm.astype('<i2').tobytes()) == measured['result']['crc32']
        tail = (-len(pcm)) % plan['v30_block_samples']
        padded = np.pad(pcm, (plan['v30_prime_samples'], tail)).astype('<i2')
        raw = features(padded)
        detector.reset(); scores = detector.predict(raw)
        detector.reset(); np.testing.assert_array_equal(scores, detector.predict(features(padded)))
        assert len(scores) == len(padded)//480-1
        quantized = np.rint(scores*256).astype(np.uint8)
        np.testing.assert_array_equal(scores, quantized.astype(np.float32)/256)
        arrays[f'output_{index}'] = quantized
        host, device = policy_result(scores, plan)
        host_rows.append({**row, 'result': host}); device_rows.append({**row, 'result': device})
        detailed.append({**row, 'prime_samples': plan['v30_prime_samples'], 'tail_zero_samples': tail,
            'frontend_shape': list(raw.shape), 'frontend_dtype': str(raw.dtype),
            'frontend_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
            'quantized_output_sha256': hashlib.sha256(quantized.tobytes()).hexdigest(),
            'output_count': len(scores), 'reset_exact': True,
            'host_convolution': host, 'firmware_arithmetic': device,
            'policies_have_identical_events': host == device})
        if index % 25 == 0:
            print(json.dumps({'completed': index+1, 'total': len(rows),
                              'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
    result = {'scope': plan['scope'], 'accepted': False,
        'plan_sha256': sha(args.plan), 'source_sha256': sha(Path(__file__)),
        'profile_sha256': sha(profile_path), 'manifest_sha256': sha(manifest),
        'baseline_sha256': sha(baseline), 'model_sha256': sha(model),
        'training_sha256': sha(args.model_dir/'training.json'), 'calibration_sha256': sha(calibration),
        'sources': len(rows), 'host_versions': plan['host_versions'],
        'v30_threshold': plan['v30_threshold'], 'smoothing_steps': plan['smoothing_steps'],
        'prime_samples': plan['v30_prime_samples'],
        'all_sources_hash_crc_verified': True, 'all_reset_replays_exact': True,
        'host_device_policy_differences': [row['path'] for row in detailed
                                         if not row['policies_have_identical_events']],
        'host_prime_events': sum(len(row['result']['prime_event_offsets']) for row in host_rows),
        'device_prime_events': sum(len(row['result']['prime_event_offsets']) for row in device_rows),
        'v30_host_convolution': groups(host_rows, 'hey_nyx'),
        'v30_firmware_arithmetic': groups(device_rows, 'hey_nyx'),
        'hiesp_retained_p4': groups(official['clips'], 'hi_esp'),
        'limits': plan['limits'], 'source_details': detailed}
    np.savez_compressed(args.output/'scores.npz', **arrays)
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'sources': len(rows), 'accepted': False,
        'v30': result['v30_host_convolution']['all'],
        'hiesp': result['hiesp_retained_p4']['all'],
        'policy_differences': len(result['host_device_policy_differences'])}), flush=True)


if __name__ == '__main__':
    main()
