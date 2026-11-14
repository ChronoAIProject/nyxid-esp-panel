"""Independently check published four-arm metadata, not saved-array inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess


def review(ref):
    ref = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    hashes = {}
    def read(name):
        path = 'research/wakeword_p4/results/' + name
        raw = subprocess.check_output(['git', 'show', ref + ':' + path])
        hashes[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    result = read('4090-hey-broad-paired-qat-screen-result-v1.json')
    audit = read('4090-hey-broad-paired-qat-screen-independent-audit-v1.json')
    lock = read('4090-hey-broad-paired-qat-screen-execution-v1.json')
    assert result['execution_sha256'] == hashes['research/wakeword_p4/results/4090-hey-broad-paired-qat-screen-execution-v1.json']
    assert audit['result_sha256'] == hashes['research/wakeword_p4/results/4090-hey-broad-paired-qat-screen-result-v1.json']
    gates = dict(float_offline_stream_max_abs=.0001, per_input_five_step_mean_abs_error_max=.02,
        per_input_five_step_p99_abs_error_max=.1, per_input_five_step_max_abs_error_max=.25)
    assert audit['numerical_gates'] == gates
    groups = defaultdict(list)
    for case in result['cases']:
        e = case['smoothed_abs_error']
        assert all(math.isfinite(v) for v in [case['phase_max_error'], *e.values()])
        passed = (case['repeat_exact'] is True and case['phase_max_error'] <= .0001
                  and e['mean'] <= .02 and e['p99'] <= .1 and e['maximum'] <= .25)
        assert passed == case['passed']
        groups[case['condition'] + '/' + case['target']].append(case)
    assert len(result['cases']) == 540 and len(groups) == 4
    summary = {}
    for key, cases in groups.items():
        assert len(cases) == len({c['request_order'] for c in cases}) == 135
        row = dict(cases=len(cases), passed=sum(c['passed'] for c in cases),
            failed_requests=sorted(c['request_order'] for c in cases if not c['passed']),
            worst_mean=max(c['smoothed_abs_error']['mean'] for c in cases),
            worst_p99=max(c['smoothed_abs_error']['p99'] for c in cases),
            worst_maximum=max(c['smoothed_abs_error']['maximum'] for c in cases))
        assert row == audit['summary'][key]
        summary[key] = row
    for target in ['hey_willow', 'hey_nyx']:
        assert {c['request_order'] for c in groups['float_control/' + target]} == {
            c['request_order'] for c in groups['integer_qat/' + target]}
    assert all(v['passed'] < 135 for v in summary.values()) and not audit['qualified_complete270_conditions']
    assert result['all_passed'] is False

    old = read('4090-hey-broad-equalized-export-result-v1.json')
    new = read('4090-hey-broad-paired-qat-export-result-v1.json')
    drift = read('4090-hey-broad-paired-qat-grid-drift-v1.json')
    checked = []
    for arm in new['arms']:
        baseline, = [a for a in old['arms'] if a['target'] == arm['target']]
        published, = [a for a in drift['arms'] if (a['target'], a['condition']) == (arm['target'], arm['condition'])]
        tensors = [{t['index']: t for t in a['tensors']} for a in [baseline, arm]]
        operations = [[o for o in a['ops'] if o['op'] in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']] for a in [baseline, arm]]
        assert [o['op'] for o in operations[0]] == [o['op'] for o in operations[1]]
        rows = []
        for number, (before, after) in enumerate(zip(*operations, strict=True)):
            a, b = [ts[o['outputs'][0]] for ts, o in zip(tensors, [before, after], strict=True)]
            row = dict(ordinal=number, operator=before['op'], old_output_index=a['index'],
                new_output_index=b['index'], old_scale=a['scale'], new_scale=b['scale'],
                new_old_scale_ratio=b['scale'] / a['scale'], old_zero=a['zero'], new_zero=b['zero'])
            rows.append(row)
        assert rows == published['activation_grid_comparison']
        checked.append(dict(target=arm['target'], condition=arm['condition'],
            model_sha256=arm['sha256'], compared_operators=len(rows),
            max_activation_scale_ratio=max(r['new_old_scale_ratio'] for r in rows)))
    return dict(schema='mac-paired-qat-published-metadata-review-v1', gpu_ref=ref,
        source_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recomputed_summary=summary, all540_phase_and_repeat_passed=all(
            c['repeat_exact'] and c['phase_max_error'] <= .0001 for c in result['cases']),
        activation_grids=checked, all_four_candidates_rejected=True, metadata_consistent=True,
        limits='Independent recomputation from published case statistics/export tensor metadata and immutable report hashes only. No access to all540 saved arrays/new final H5/TFLite bytes, no actual new inference or unique-cause claim. Not a replacement for GPU saved-array audit, numerical admission, recognition, freshdev, independent acceptance or P4 validation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.gpu_ref)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(metadata_consistent=True, all_four_candidates_rejected=True,
                         compared_operators=sum(a['compared_operators'] for a in result['activation_grids']))))
