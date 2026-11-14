"""Compare four saved native Mac witnesses with authenticated GPU originals.

No inference, tolerance selection, training, new development or audio IO.
Float cross-platform differences are descriptive, not a new admission gate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(scores, floats, scale, zero):
    signed = (scores.astype(np.float64) - zero) * scale - floats
    error = np.abs(np.convolve(signed, np.ones(5) / 5, mode='valid'))
    return dict(mean=float(error.mean()), p99=float(np.quantile(error, .99)),
                maximum=float(error.max()))


def run(a):
    base = Path(__file__).resolve().parents[1]
    plan_path = base / 'results/mac-bounded-consistency-fixed-grid-witness-replay-plan-v1.json'
    plan = json.loads(plan_path.read_text())
    local_path = a.replay_root / 'report.json'
    local = json.loads(local_path.read_text())
    assert local['plan_sha256'] == sha(plan_path)
    assert local['script_sha256'] == plan['script_sha256']
    manifest = json.loads((a.received / 'manifest.json').read_text())
    for item in manifest['files']:
        path = a.received / item['name']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    gpu_path = a.received / 'metadata/4090-hey-broad-bounded-consistency-fixed-grid-screen-result-v1.json'
    gpu = json.loads(gpu_path.read_text())
    results = []
    for arm in plan['arms']:
        target, condition = arm['target'], arm['condition']
        row, = [r for r in local['cases'] if
                (r['target'], r['condition']) == (target, condition)]
        request = next(i['request_order'] for i in plan['inputs'] if i['target'] == target)
        assert row['request_order'] == request
        reference, = [r for r in gpu['cases'] if
                      (r['target'], r['condition'], r['request_order']) ==
                      (target, condition, request)]
        assert row['model_sha256'] == arm['model_sha256']
        assert row['final_h5_sha256'] == arm['weights_sha256']
        lp = a.replay_root / row['arrays']
        gp = a.received / ('witnesses/' + target + '-' + condition + '-request' + str(request) + '.npz')
        assert sha(lp) == row['arrays_sha256']
        assert sha(gp) == reference['arrays_sha256']
        with np.load(lp, allow_pickle=False) as data:
            left = {key: data[key].copy() for key in data.files}
        with np.load(gp, allow_pickle=False) as data:
            right = {key: data[key].copy() for key in data.files}
        for data in [left, right]:
            assert all(np.isfinite(value).all() for value in data.values())
        exact_fields = {key: bool(np.array_equal(left[key], right[key]))
                        for key in ['raw', 'quantized_input', 'uint8']}
        repeats = {key: bool(np.array_equal(left[key], left[key + '_repeat']))
                   for key in ['float', 'uint8', 'embedding', 'actual_fc']}
        registered = bool(np.array_equal(left['actual_fc'], left['registered_fc']))
        assert len(left['float']) == row['steps'] == len(right['float'])
        scale, zero = reference['output_quantization']
        local_metrics = metrics(left['uint8'], left['float'], scale, zero)
        gpu_metrics = metrics(right['uint8'], right['float'], scale, zero)
        # Reconstruct each report from its own saved arrays. This comparison
        # allowance covers JSON/float64 metadata only, not float-vs-int8 gates.
        for calculated, reported in [(local_metrics, row['smoothed_abs_error']),
                                     (gpu_metrics, reference['smoothed_abs_error'])]:
            assert all(np.isclose(calculated[k], reported[k], rtol=1e-12, atol=1e-15)
                       for k in calculated)
        phase = float(np.max(np.abs(left['offline_phase'] - left['float'][74:])))
        assert phase == row['float_offline_phase_max_abs']
        gate = phase <= 1e-4 and local_metrics['mean'] <= .02 and local_metrics['p99'] <= .1 and local_metrics['maximum'] <= .25
        gate_matches = gate == row['witness_meets_numerical_bounds'] == reference['passed']
        passed = all(exact_fields.values()) and all(repeats.values()) and registered and gate_matches
        results.append(dict(target=target, condition=condition, request_order=request,
            steps=row['steps'], cross_platform_exact=exact_fields, native_repeats_exact=repeats,
            registered_fc_exact=registered,
            cross_platform_float_max_abs=float(np.max(np.abs(left['float'] - right['float']))),
            cross_platform_phase_max_abs=float(np.max(np.abs(left['offline_phase'] - right['offline_phase']))),
            original_numerical_gate=gate, original_gate_outcome_matches_gpu=gate_matches,
            local_metrics=local_metrics, gpu_metrics=gpu_metrics,
            local_arrays_sha256=sha(lp), gpu_arrays_sha256=sha(gp), audit_passed=passed))
    assert len(results) == len(local['cases']) == 4
    report = dict(schema='mac-bounded-consistency-fixed-grid-saved-array-cross-platform-audit-v1',
        script_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path),
        native_report_sha256=sha(local_path), gpu_report_sha256=sha(gpu_path),
        cases=results, total_steps=sum(r['steps'] for r in results),
        all_passed=all(r['audit_passed'] for r in results),
        limits='Four predetermined old witnesses only. Actual integer outputs and raw/quantized inputs compared against authenticated GPU NPZ, native repeats and FC registration checked. Float cross-platform maxima are descriptive; original numerical gate outcomes remain unchanged. Not all540/P4 inference, recognition, FAR, fresh-development or independent acceptance.')
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    assert report['all_passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--received', type=Path, required=True)
    parser.add_argument('--replay-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
