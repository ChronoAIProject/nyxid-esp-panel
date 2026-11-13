"""Mac crosscheck of published quantization reports, not private model/array IO."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess


def review(ref):
    base = 'research/wakeword_p4/results/'
    hashes = {}

    def read(name):
        raw = subprocess.check_output(['git', 'show', ref+':'+base+name])
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    plan = read('4090-hey-broad-trained-quantization-plan-v1.json')
    execution = read('4090-hey-broad-trained-quantization-execution-v1.json')
    original = read('4090-hey-broad-trained-quantization-result-v1.json')
    independent = read('4090-hey-broad-trained-quantization-independent-array-audit-v1.json')
    rounded = read('4090-hey-broad-input-rounding-result-v1.json')
    rounded_summary = read('4090-hey-broad-input-rounding-summary-v1.json')
    lattice = read('4090-hey-broad-quantization-lattice-audit-v1.json')
    assert original['execution_sha256'] == hashes['4090-hey-broad-trained-quantization-execution-v1.json']
    assert independent['original_result_sha256'] == lattice['original_result_sha256'] == hashes['4090-hey-broad-trained-quantization-result-v1.json']
    assert lattice['original_execution_sha256'] == original['execution_sha256']
    assert rounded_summary['original_report_sha256'] == hashes['4090-hey-broad-input-rounding-result-v1.json']
    requests = {row['order'] for row in plan['requests']}
    assert len(requests) == len(plan['requests']) == 135
    expected = {(target, order) for target in ('hey_willow', 'hey_nyx') for order in requests}
    rows = {(row['target'], row['request_order']): row for row in original['cases']}
    round_rows = {(row['target'], row['request_order']): row for row in rounded['cases']}
    assert len(rows) == len(original['cases']) == len(round_rows) == len(rounded['cases']) == 270
    assert set(rows) == set(round_rows) == expected
    gates = plan['numerical_gates']
    failures = Counter()
    targets = {}
    for target in ('hey_willow', 'hey_nyx'):
        training = read('4090-hey-broad-formal-'+target.removeprefix('hey_')+'-training-v1.json')
        receipt = read('4090-hey-broad-formal-'+target.removeprefix('hey_')+'-execution-receipt-v1.json')
        report_name = '4090-hey-broad-formal-'+target.removeprefix('hey_')+'-training-v1.json'
        model = lattice['models'][target]
        assert training['model_sha256'] == receipt['model_sha256'] == model['model_sha256']
        assert training['model_bytes'] == model['model_bytes'] == 57592
        assert receipt['training_report_sha256'] == hashes[report_name]
        assert training['steps'] == training['selected_step'] == receipt['steps'] == 120000
        assert training['batch_schedule']['verified_batches'] == 120100
        assert training['batch_schedule']['all_actual_batch_tensors_match_precomputed'] is True
        assert training['batchnorm_initial'] == training['batchnorm_selected']
        assert all(receipt[k] is True for k in ('finite_weights', 'finite_logged_loss', 'frozen_bn_exact'))
        group = [row for (name, _), row in rows.items() if name == target]
        for row in group:
            smooth = row['smoothed_abs_error']
            checks = {'smoothed_mae': smooth['mean'] <= gates['per_input_five_step_mean_abs_error_max'],
                'smoothed_p99': smooth['p99'] <= gates['per_input_five_step_p99_abs_error_max'],
                'smoothed_max': smooth['maximum'] <= gates['per_input_five_step_max_abs_error_max']}
            assert all(math.isfinite(v) for v in smooth.values())
            passed = row['repeat_exact'] and row['phase_max_error'] <= gates['float_offline_stream_max_abs'] and all(checks.values())
            assert bool(passed) == row['passed']
            for name, value in checks.items():
                if not value:
                    failures[target+'|'+name] += 1
            rounded_row = round_rows[(target, row['request_order'])]
            assert rounded_row['twice_reset_exact'] is True
            for left, right in [('mean', 'mean_absolute'), ('p99', 'p99_absolute'), ('maximum', 'maximum_absolute')]:
                assert math.isclose(smooth[left], rounded_row['total_original'][right], rel_tol=0, abs_tol=1e-12)
        # Independent scalar calculation over every possible INT8 logit code.
        codes = set()
        for code in range(-128, 128):
            value = (code-model['logit']['zero_point'])*model['logit']['scale']
            if value >= 0:
                probability = 1/(1+math.exp(-value))
            else:
                exp_value = math.exp(value)
                probability = exp_value/(1+exp_value)
            codes.add(min(255, max(0, round(probability*256))))
        assert sorted(codes) == model['ideal_sigmoid_output_codes']
        assert {int(code) for code in model['actual_saved_output_code_counts']} <= codes
        assert sum(model['actual_saved_output_code_counts'].values()) == model['raw_outputs']
        for state in model['states']:
            assert all(state['read'][key] == state['assigned'][key] for key in ('shape', 'dtype', 'scale', 'zero_point'))
        assert len(model['states']) == 6
        rounded_group = [row for (name, _), row in round_rows.items() if name == target]
        maxima = {}
        for kind in ('input_rounding', 'remaining_quantization', 'total_original'):
            observed = dict(worst=max(r[kind]['maximum_absolute'] for r in rounded_group),
                p99_max=max(r[kind]['p99_absolute'] for r in rounded_group),
                mean_max=max(r[kind]['mean_absolute'] for r in rounded_group))
            assert observed == rounded_summary['summary'][target][kind]
            maxima[kind] = observed['worst']
        targets[target] = dict(cases=len(group), passed=sum(row['passed'] for row in group),
            phase_max=max(row['phase_max_error'] for row in group),
            all_repeat_exact=all(row['repeat_exact'] for row in group),
            logit_step=model['logit']['scale'], mathematical_output_codes=sorted(codes),
            reported_output_code_counts=model['actual_saved_output_code_counts'], error_maxima=maxima)
    assert dict(failures) == independent['failed_gate_counts']
    assert original['all_passed'] is False
    return dict(schema='mac-broad-quantization-metadata-review-v1', gpu_ref=ref, artifact_sha256=hashes,
        case_identities_and_frozen_gate_recalculation_exact=True,
        rounded_original_error_and_summary_crosscheck_exact=True,
        scalar_logit_lattice_calculation_exact=True, targets=targets, failed_gates=dict(failures),
        current_exports_rejected=True, private_arrays_or_models_independently_read=False,
        limits='Crosschecks published metadata and independently calculates scalar logistic lattice only. GPU saved-array/model/state claims are not independently rerun on Mac. No proof of calibration range causality, dynamic state correctness, recognition or P4 acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.gpu_ref)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
