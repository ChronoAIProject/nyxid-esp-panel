"""Replay a byte-pinned candidate on all 350 existing Mac regression sources.

No calibration, training, source selection, audio I/O or device access occurs.
Freeze the candidate profile before scoring. The legacy v30 study is unchanged.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
import time
import zlib

import numpy as np
import soundfile as sf

from evaluate import Detector
from features import features
from prepare_runtime_selftest import inspect_model
from replay_gpu_mac_regression import groups, policy_result


SOURCE_PLAN_SHA = 'c7569b5eceef7d3204ed4a8622dec32a41519bda4766764d15237941ae078442'
REFERENCE_SHA = '47ccac0cfd8d9b3809aa62e6b9f8232e70ea1d12655959648a4067b3a608c57b'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'model-dir', 'profile', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-scores', type=Path,
                        help='Only for the predeclared legacy-v30 runner regression')
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output directory required')
    profile = json.loads(args.profile.read_bytes())
    require(re.fullmatch(r'[a-zA-Z0-9_.-]{1,80}', profile['candidate_id']), 'Invalid candidate ID')
    source_plan = Path(__file__).parent/'plans/gpu-v30-mac-regression-v1.json'
    require(sha(source_plan) == SOURCE_PLAN_SHA, 'Original source protocol changed')
    plan = json.loads(source_plan.read_bytes())
    require(sha(args.reference) == REFERENCE_SHA, 'Original full Mac reference changed')
    reference = json.loads(args.reference.read_bytes())
    for key, expected in plan['host_versions'].items():
        require(version(key) == expected, 'Unreviewed host version: '+key)
    require(profile['source_plan_sha256'] == SOURCE_PLAN_SHA
            and profile['reference_sha256'] == REFERENCE_SHA, 'Profile input protocol differs')
    model, training_path = args.model_dir/'hey_nyx.tflite', args.model_dir/'training.json'
    calibration_name = Path(profile['calibration_file'])
    require(not calibration_name.is_absolute() and '..' not in calibration_name.parts,
            'Calibration must be relative to the candidate directory')
    calibration_path = args.model_dir/calibration_name
    for path, field in [(model, 'model_sha256'), (training_path, 'training_report_sha256'),
                        (calibration_path, 'calibration_sha256')]:
        require(sha(path) == profile[field], 'Candidate bytes differ: '+path.name)
    training = json.loads(training_path.read_bytes())
    calibration = json.loads(calibration_path.read_bytes())
    require(training['model_sha256'] == calibration['model_sha256'] == profile['model_sha256'],
            'Model, training and calibration identities differ')
    require(training['architecture'] == 'teacher60' and training['context_frames'] == 150,
            'This protocol supports the reviewed teacher60/context150 geometry only')
    require(calibration['threshold'] == profile['threshold']
            and calibration['sliding_window'] == profile['smoothing_steps'] == 5
            and profile['refractory_steps'] == 67 and 0 < profile['threshold'] < 1,
            'Candidate must retain its frozen five-score/67-step calibration')
    require(profile['prime_samples'] == plan['v30_prime_samples'] == 56160,
            'Application silence prime changed')
    plan = {**plan, 'v30_threshold': profile['threshold']}
    manifest, baseline_path = args.sources/'manifest.jsonl', args.sources/'baseline.json'
    require(sha(manifest) == plan['manifest_sha256']
            and sha(baseline_path) == plan['baseline_sha256'], 'Original paired inputs changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    official = json.loads(baseline_path.read_bytes())
    require(len(rows) == len(official['clips']) == len(reference['source_details']) == 350,
            'Every original source is required')
    require(dict(Counter(row['target'] for row in rows)) == plan['target_counts']
            and sorted({r['voice'] for r in rows}) == plan['voices']
            and sorted({r['variant'] for r in rows}) == plan['variants'], 'Source cohorts differ')
    require(official['restored'] and official['manifest_sha256'] == sha(manifest),
            'Original Hi ESP baseline provenance differs')
    gold = None
    if args.reference_scores:
        require(profile['model_sha256'] == reference['model_sha256']
                and profile['threshold'] == reference['v30_threshold'],
                'Reference-output check is only valid for the original model/policy')
        require(sha(args.reference_scores) == profile['reference_scores_sha256'],
                'Legacy saved outputs changed')
        gold = np.load(args.reference_scores, allow_pickle=False)
        require(set(gold.files) == {f'output_{i}' for i in range(350)}, 'Incomplete legacy outputs')
    else:
        require('reference_scores_sha256' not in profile, 'Declared reference outputs were omitted')
    detector = Detector(model)
    interface = inspect_model(model.read_bytes(), detector)
    # Authenticate every source before any model scoring, with no successful subset.
    for row, measured, prior in zip(rows, official['clips'], reference['source_details'], strict=True):
        require(all(measured[k] == prior[k] == v for k, v in row.items()), 'Source order/identity differs')
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        require(sha(args.sources/relative) == row['sha256'] and measured['result']['complete'],
                'Source WAV bytes or official completion differ')
    args.output.mkdir(parents=True)
    for path, name in [(args.profile, 'profile.json'), (source_plan, 'source-plan.json'),
                       (Path(__file__), 'source.py')]:
        shutil.copyfile(path, args.output/name)
    arrays, details, host_rows, device_rows = {}, [], [], []
    event_plan = {**plan, 'v30_threshold': float(np.float32(profile['threshold']))}
    started = time.monotonic()
    for index, (row, measured, prior) in enumerate(zip(rows, official['clips'], reference['source_details'], strict=True)):
        pcm, rate = sf.read(args.sources/row['path'], dtype='int16')
        require(rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples'], 'PCM geometry differs')
        require(zlib.crc32(pcm.astype('<i2').tobytes()) == measured['result']['crc32'], 'PCM CRC differs')
        tail = (-len(pcm)) % 480
        raw = features(np.pad(pcm, (56160, tail)).astype('<i2'))
        raw_sha = hashlib.sha256(raw.tobytes()).hexdigest()
        require(raw_sha == prior['frontend_sha256'] and list(raw.shape) == prior['frontend_shape']
                and str(raw.dtype) == prior['frontend_dtype'], 'Full original frontend array differs')
        detector.reset()
        scores = detector.predict(raw)
        detector.reset()
        np.testing.assert_array_equal(scores, detector.predict(raw))
        quantized = np.rint(scores*256).astype(np.uint8)
        np.testing.assert_array_equal(scores, quantized.astype(np.float32)/256)
        require(len(scores) == prior['output_count'], 'Output geometry differs')
        if gold is not None:
            np.testing.assert_array_equal(quantized, gold[f'output_{index}'])
        host, _ = policy_result(scores, plan)
        _, device = policy_result(scores, event_plan)
        if gold is not None:
            require(host == prior['host_convolution'] and device == prior['firmware_arithmetic'],
                    'Legacy event replay differs')
        arrays[f'output_{index}'] = quantized
        host_rows.append({**row, 'result': host})
        device_rows.append({**row, 'result': device})
        details.append({**row, 'frontend_sha256': raw_sha,
                        'output_sha256': hashlib.sha256(quantized.tobytes()).hexdigest(),
                        'output_count': len(scores), 'prime_samples': 56160, 'tail_zero_samples': tail,
                        'reset_exact': True, 'host_convolution': host, 'firmware_arithmetic': device})
        if index % 50 == 0:
            print(json.dumps(dict(completed=index+1, total=350,
                                  elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
    summaries = dict(candidate_host=groups(host_rows, 'hey_nyx'),
                     candidate_firmware=groups(device_rows, 'hey_nyx'),
                     hiesp_retained_p4=groups(official['clips'], 'hi_esp'))
    if gold is not None:
        require(summaries['candidate_host'] == reference['v30_host_convolution']
                and summaries['candidate_firmware'] == reference['v30_firmware_arithmetic'],
                'Legacy aggregate/cohort results differ')
    np.savez_compressed(args.output/'scores.npz', **arrays)
    helper_paths = [Path(__file__).with_name(name+'.py') for name in (
        'evaluate', 'features', 'detector_policy', 'prepare_runtime_selftest', 'replay_gpu_mac_regression')]
    helper_paths.append(Path(__file__).resolve().parents[2]/'wakeword_baseline/compare.py')
    result = dict(scope='Complete reused Mac regression, fixed candidate profile; not new acceptance.',
                  accepted=False, candidate_id=profile['candidate_id'], profile_sha256=sha(args.profile),
                  model_sha256=sha(model), training_sha256=sha(training_path),
                  calibration_sha256=sha(calibration_path), source_sha256=sha(Path(__file__)),
                  helper_source_sha256={str(p.relative_to(Path(__file__).resolve().parents[2])): sha(p)
                                        for p in helper_paths},
                  reference_sha256=sha(args.reference), source_plan_sha256=sha(source_plan),
                  manifest_sha256=sha(manifest), baseline_sha256=sha(baseline_path),
                  interface=interface, host_versions=plan['host_versions'],
                  threshold=profile['threshold'], firmware_threshold=float(np.float32(profile['threshold'])),
                  smoothing_steps=5, refractory_steps=67, sources=350,
                  quantized_outputs=sum(row['output_count'] for row in details),
                  all_frontend_arrays_match_original=True, all_reset_replays_exact=True,
                  legacy_saved_outputs_and_events_exact=gold is not None,
                  scores_sha256=sha(args.output/'scores.npz'), summaries=summaries,
                  source_details=details, limits=[
                      'Previously consumed correlated synthetic sources; no phonetic/human recall or matched-FAR claim.',
                      'Candidate is host INT8 reference; Hi ESP uses retained P4 results. Not a candidate P4 acoustic test.',
                      'Original PCM speed and full streams retained; only original zero prime and final block padding added.',
                      'Source offsets are processing positions, not verified phonetic endpoints or acoustic latency.',
                      'No threshold selection, training, new/final audio, playback, microphone, service or board operation.'])
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('candidate_id', 'sources', 'quantized_outputs',
                                                  'legacy_saved_outputs_and_events_exact')}, indent=2))


if __name__ == '__main__':
    main()
