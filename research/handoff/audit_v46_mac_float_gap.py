"""Reconstruct saved float/INT8 classifications and zero-cohort score ranges.

No model inference. Uses math.fsum windows and independent event accounting,
without importing the producer's smoothing or detector policy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wakeword_p4/training'))
from verify_gpu_mac_regression import trigger_steps, verify_group


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('float-study', 'int8-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.float_study
    report = json.loads((root/'result.json').read_bytes())
    plan = json.loads((root/'plan.json').read_bytes())
    prior = json.loads((args.int8_study/'result.json').read_bytes())
    assert sha(root/'plan.json') == report['plan_sha256']
    assert sha(root/'source.py') == report['source_sha256']
    assert sha(args.int8_study/'result.json') == report['int8_result_sha256'] == plan['int8_result_sha256']
    assert sha(args.int8_study/'scores.npz') == report['int8_scores_sha256'] == plan['int8_scores_sha256']
    assert sha(root/'float-scores.npz') == report['float_scores_sha256']
    floats = np.load(root/'float-scores.npz', allow_pickle=False)
    quantized = np.load(args.int8_study/'scores.npz', allow_pickle=False)
    modes = ('original', 'dequantized')
    assert set(floats.files) == {f'{mode}_{i}' for mode in modes for i in range(350)}
    assert len(report['source_details']) == len(prior['source_details']) == 350
    rows = {mode: [] for mode in modes}
    mismatches = {mode: [] for mode in modes}
    maximum_residual = {mode: 0. for mode in modes}
    zero_peak = {mode: [] for mode in modes}
    maximum_input_rounding_change, checked = 0., 0
    for index, (record, old) in enumerate(zip(report['source_details'], prior['source_details'], strict=True)):
        assert all(record[k] == old[k] for k in ('path', 'sha256', 'voice', 'variant', 'text', 'target', 'frontend_sha256'))
        q = quantized[f'output_{index}']
        assert q.dtype == np.uint8 and hashlib.sha256(q.tobytes()).hexdigest() == old['output_sha256']
        windows = np.lib.stride_tricks.sliding_window_view(q, 5)
        ends = np.arange(6, len(q)+2)*480-56160
        zero = record['target'] == 'hey_nyx' and np.max(windows.sum(axis=1)[ends > 0]) == 0
        original = floats[f'original_{index}']
        dequantized = floats[f'dequantized_{index}']
        maximum_input_rounding_change = max(maximum_input_rounding_change, float(np.max(np.abs(original-dequantized))))
        for mode in modes:
            values = floats[f'{mode}_{index}']
            saved = record['modes'][mode]
            assert values.dtype == np.float32 and values.ndim == 1 and len(values) == len(q)
            assert hashlib.sha256(values.tobytes()).hexdigest() == saved['output_sha256']
            assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
            means = np.array([math.fsum(float(x) for x in values[at:at+5])/5
                              for at in range(len(values)-4)])
            steps = trigger_steps(means, plan['threshold'])
            offsets = [(step+1)*480-56160 for step in steps]
            events = dict(events=sum(at > 0 for at in offsets), offsets=[at for at in offsets if at > 0],
                          one_based_score_steps=steps, prime_event_offsets=[at for at in offsets if at <= 0])
            assert events == saved['fixed_threshold_events']
            assert abs(float(means[ends > 0].max())-saved['max_post_prime_five_mean']) <= 2e-15
            peak = float(values[(np.arange(2, len(values)+2)*480-56160) > 0].max())
            assert peak == saved['max_post_prime_score']
            maximum_residual[mode] = max(maximum_residual[mode], float(np.max(np.abs(values.astype(np.float64)-q/256))))
            if zero:
                zero_peak[mode].append(peak)
            if bool(events['events']) != bool(old['host_convolution']['events']):
                mismatches[mode].append(record['path'])
            rows[mode].append({**record, 'result': events})
            checked += len(values)
    for mode in modes:
        summary = report['summaries'][mode]
        verify_group(rows[mode], 'hey_nyx', summary['all'])
        for cohort, counts in summary['by_voice_and_variant'].items():
            voice, variant = cohort.split('/variant-')
            verify_group([r for r in rows[mode] if r['voice'] == voice and r['variant'] == int(variant)],
                         'hey_nyx', counts)
    result = dict(scope=__doc__, passed=True, accepted=False,
                  float_result_sha256=sha(root/'result.json'), float_scores_sha256=sha(root/'float-scores.npz'),
                  int8_result_sha256=sha(args.int8_study/'result.json'),
                  auditor_sha256=sha(Path(__file__)), verified_float_outputs=checked,
                  fixed_threshold_classification_differences=mismatches,
                  maximum_abs_probability_difference_from_int8=maximum_residual,
                  maximum_input_rounding_probability_change=maximum_input_rounding_change,
                  zero_target_float_peak_ranges={mode: dict(count=len(peaks), minimum=min(peaks), maximum=max(peaks),
                      below_half_uint8_output_step=sum(value < 1/512 for value in peaks)) for mode, peaks in zero_peak.items()},
                  limits='Exact saved-array/hash/event accounting on reused synthetic sources. No independent model inference or proof that export differences are zero; maximum residuals are retained. No training-cause, population FAR/recall or physical acceptance claim.')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
