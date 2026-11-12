"""Reconstruct a candidate's Mac regression from saved UINT8 outputs.

Uses an exact-rational integer firmware policy, without importing the producer's
detector/event helpers. Recomputes frontend bytes and all cohort event counts;
does not independently infer a model or prove pronunciation/acoustic quality.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import zlib

import numpy as np
import soundfile as sf

from features import features
from verify_gpu_mac_regression import trigger_steps, verify_group


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'artifacts', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh audit output required')
    root = args.artifacts
    result = json.loads((root/'result.json').read_bytes())
    profile = json.loads((root/'profile.json').read_bytes())
    reference = json.loads(args.reference.read_bytes())
    plan = json.loads((root/'source-plan.json').read_bytes())
    for path, digest in [(root/'profile.json', result['profile_sha256']),
                         (root/'source-plan.json', result['source_plan_sha256']),
                         (root/'source.py', result['source_sha256']),
                         (root/'scores.npz', result['scores_sha256']),
                         (args.reference, profile['reference_sha256']),
                         (args.sources/'manifest.jsonl', result['manifest_sha256']),
                         (args.sources/'baseline.json', result['baseline_sha256'])]:
        require(sha(path) == digest, 'Input bytes differ: '+path.name)
    require(result['accepted'] is False and result['all_reset_replays_exact'],
            'Unexpected acceptance/reset report')
    require(result['source_plan_sha256'] == profile['source_plan_sha256']
            and result['model_sha256'] == profile['model_sha256']
            and result['training_sha256'] == profile['training_report_sha256']
            and result['calibration_sha256'] == profile['calibration_sha256'],
            'Frozen candidate/profile identities differ')
    require(result['threshold'] == profile['threshold']
            and result['firmware_threshold'] == float(np.float32(profile['threshold']))
            and profile['smoothing_steps'] == result['smoothing_steps'] == 5
            and profile['refractory_steps'] == result['refractory_steps'] == 67,
            'Event policy differs')
    manifest = [json.loads(line) for line in (args.sources/'manifest.jsonl').read_text().splitlines()]
    baseline = json.loads((args.sources/'baseline.json').read_bytes())
    require(len(manifest) == len(baseline['clips']) == len(reference['source_details'])
            == len(result['source_details']) == 350, 'Incomplete original sources')
    saved = np.load(root/'scores.npz', allow_pickle=False)
    require(set(saved.files) == {f'output_{i}' for i in range(350)}, 'Incomplete score arrays')
    bound = Fraction.from_float(result['firmware_threshold'])*1280
    host_rows, device_rows, differences = [], [], []
    outputs = 0
    for index, (item, official, prior, row) in enumerate(zip(
            manifest, baseline['clips'], reference['source_details'], result['source_details'], strict=True)):
        require(all(official[k] == prior[k] == row[k] == v for k, v in item.items()),
                'Original source order/identity differs')
        path = args.sources/item['path']
        require(sha(path) == item['sha256'], 'Source WAV changed')
        pcm, rate = sf.read(path, dtype='int16')
        require(rate == 16000 and pcm.ndim == 1 and len(pcm) == item['samples']
                and zlib.crc32(pcm.astype('<i2').tobytes()) == official['result']['crc32'],
                'Original PCM differs')
        tail = (-len(pcm)) % 480
        require(row['prime_samples'] == profile['prime_samples'] == plan['v30_prime_samples'] == 56160
                and row['tail_zero_samples'] == tail, 'Prime/padding differs')
        front = features(np.pad(pcm, (56160, tail)).astype('<i2'))
        require(hashlib.sha256(front.tobytes()).hexdigest() == row['frontend_sha256']
                == prior['frontend_sha256'] and list(front.shape) == prior['frontend_shape']
                and str(front.dtype) == prior['frontend_dtype'], 'Original frontend array differs')
        q = saved[f'output_{index}']
        require(q.dtype == np.uint8 and q.ndim == 1
                and len(q) == row['output_count'] == prior['output_count']
                == (56160+len(pcm)+tail)//480-1
                and hashlib.sha256(q.tobytes()).hexdigest() == row['output_sha256'],
                'Saved UINT8 outputs differ')
        totals = [sum(map(int, q[at:at+5]))*bound.denominator for at in range(len(q)-4)]
        device = trigger_steps(totals, bound.numerator)
        means = np.convolve(q.astype(np.float64)/256, np.ones(5)/5, mode='valid')
        host = trigger_steps(means, result['threshold'])
        for key, steps in [('host_convolution', host), ('firmware_arithmetic', device)]:
            offsets = [(step+1)*480-56160 for step in steps]
            require(row[key] == dict(events=sum(at > 0 for at in offsets),
                    offsets=[at for at in offsets if at > 0], one_based_score_steps=steps,
                    prime_event_offsets=[at for at in offsets if at <= 0]),
                    'Saved event reconstruction differs: '+item['path'])
        if host != device:
            differences.append(item['path'])
        host_rows.append({**item, 'result': row['host_convolution']})
        device_rows.append({**item, 'result': row['firmware_arithmetic']})
        outputs += len(q)
    for key, rows, target in [('candidate_host', host_rows, 'hey_nyx'),
                               ('candidate_firmware', device_rows, 'hey_nyx'),
                               ('hiesp_retained_p4', baseline['clips'], 'hi_esp')]:
        summary = result['summaries'][key]
        require(set(summary['by_voice_and_variant']) == {
            f"{row['voice']}/variant-{row['variant']}" for row in manifest}, 'Incomplete cohorts')
        verify_group(rows, target, summary['all'])
        for cohort, counts in summary['by_voice_and_variant'].items():
            voice, variant = cohort.split('/variant-')
            verify_group([r for r in rows if r['voice'] == voice and r['variant'] == int(variant)],
                         target, counts)
    audit = dict(scope=__doc__, passed=True, accepted=False, candidate_id=profile['candidate_id'],
                 result_sha256=sha(root/'result.json'), profile_sha256=sha(root/'profile.json'),
                 scores_sha256=sha(root/'scores.npz'), auditor_sha256=sha(Path(__file__)),
                 source_count=350, uint8_outputs_verified=outputs,
                 exact_firmware_threshold_sum=str(bound), host_device_event_differences=differences,
                 all_original_pcm_crcs_and_frontend_hashes_match=True,
                 all_saved_events_and_cohort_counts_reconstruct=True,
                 accounting_helper_sha256=sha(Path(__file__).with_name('verify_gpu_mac_regression.py')),
                 limits='Saved-output and frontend accounting only; no independent model inference, reset rerun, new/final data, audio, board or quality acceptance.')
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    shutil.copyfile(__file__, root/'auditor-source.py')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
