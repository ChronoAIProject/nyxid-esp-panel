"""Exercise the new table auditor on real old tables and labelled parser fixtures."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import audit_paired_diagnostic_tables_mac_v1 as checker


def review_poisson():
    import scipy
    from scipy.stats import chi2
    rows = []
    for count in list(range(100)) + [100, 101, 200, 1000, 10000, 100000, 1000000]:
        direct = checker.poisson_upper95(count, 10.483944826388889)
        reference = float(.5 * chi2.ppf(.95, 2 * (count + 1)) / 10.483944826388889)
        error = abs(direct - reference)
        assert error <= 2e-10 * max(1., reference)
        rows.append(dict(count=count, direct_upper95=direct, reference_upper95=reference,
                         absolute_error=error, relative_error=error / reference))
    return dict(scipy_version=scipy.__version__, cases=rows,
                maximum_absolute_error=max(r['absolute_error'] for r in rows),
                maximum_relative_error=max(r['relative_error'] for r in rows))


def review(ref):
    commit = subprocess.check_output(['git', 'rev-parse', ref], text=True).strip()
    inputs = {}

    def read(name):
        path = checker.BASE + name
        data = subprocess.check_output(['git', 'show', commit + ':' + path])
        inputs[path] = checker.sha(data)
        return json.loads(data)

    prefix = checker.PROFILES['raw-context'][0]
    old = dict(table=read(prefix + 'diagnostic-summary-v1.json'),
               grouped=read(prefix + 'grouped-diagnostic-v1.json'),
               audit=read(prefix + 'scoring-independent-audit-v1.json'),
               execution=read(prefix + 'development-execution-v1.json'),
               original=read('4090-fp32-reused-development-execution-v1.json'), inputs=inputs)
    baseline = checker.audit_loaded('raw-context', **old)
    assert baseline['checked_points'] == 552
    assert [r['joint_qualified_points'] for r in baseline['recomputed_summary']] == [0] * 4

    # Rename old labels only to exercise the future profile's schema. These are
    # not new weights, scores, traces, training or recognition results.
    def rename(data):
        return json.loads(json.dumps(data).replace('-command_context', '-frozen').replace('-control', '-full'))

    fixture = rename(old)
    groups, audit = fixture['grouped'], fixture['audit']
    table = dict(all_new_points=rename(old['table']['points']),
                 selected_threshold=None, candidate_promoted=False,
                 files={'.fixture/frozen/diagnostic/summary.json': audit['summary_sha256']},
                 summary=[], all_matched_deltas=[], rejected_for_no_joint_point=[])
    for row in rename(baseline['recomputed_summary']):
        row = dict(row)
        target, condition = row.pop('model').rsplit('-', 1)
        table['summary'].append(dict(target=target, condition=condition, **row))
        if row['joint_qualified_points'] == 0:
            table['rejected_for_no_joint_point'].append(target + '-' + condition)
    for pair in baseline['all_matched_points']:
        d = pair['second_minus_first']
        table['all_matched_deltas'].append(dict(target=pair['target'], threshold=pair['threshold'],
            frozen_minus_full_complete=d['complete'], frozen_minus_full_background_events=d['background_events'],
            frozen_minus_full_pre_source=d['pre_source'], frozen_minus_full_common_events=d['common_events'],
            frozen_minus_full_cross_target_events=d['cross_events']))
    new_prefix = checker.PROFILES['frozen-encoder'][0]
    fixture['table'] = table
    fixture['inputs'] = {
        checker.BASE + new_prefix + 'scoring-independent-audit-v1.json': checker.sha(json.dumps(audit).encode()),
        checker.BASE + new_prefix + 'development-execution-v1.json': checker.sha(json.dumps(fixture['execution']).encode())}
    groups['scoring_audit_sha256'] = fixture['inputs'][checker.BASE + new_prefix + 'scoring-independent-audit-v1.json']
    converted = checker.audit_loaded('frozen-encoder', **fixture)
    assert converted['recomputed_summary'] == rename(baseline['recomputed_summary'])
    assert converted['all_matched_points'] == baseline['all_matched_points']

    def wrong_group_count(d):
        d['grouped']['models'][0]['points'][0]['groups'][0]['clips'] += 1

    def extra_dimension(d):
        g = copy.deepcopy(d['grouped']['models'][0]['points'][0]['groups'][0])
        g['dimension'] = 'invented'
        d['grouped']['models'][0]['points'][0]['groups'].append(g)

    def wrong_far(d):
        d['table']['all_new_points'][0]['background']['far_upper95'] += 1
        d['grouped']['models'][0]['points'][0]['background']['far_upper95'] += 1

    mutations = {
        'missing_point': lambda d: d['table']['all_new_points'].pop(),
        'duplicate_point': lambda d: d['table']['all_new_points'].__setitem__(1, copy.deepcopy(d['table']['all_new_points'][0])),
        'missing_model': lambda d: d['grouped']['models'].pop(),
        'wrong_group_count': wrong_group_count,
        'extra_group_dimension': extra_dimension,
        'wrong_poisson_bound': wrong_far,
        'wrong_audit_binding': lambda d: d['inputs'].__setitem__(checker.BASE + new_prefix + 'scoring-independent-audit-v1.json', '0' * 64),
        'wrong_declared_summary': lambda d: d['table']['summary'][0].__setitem__('far_qualified_points', 1),
        'wrong_declared_delta': lambda d: d['table']['all_matched_deltas'][0].__setitem__('frozen_minus_full_common_events', 999),
        'wrong_rejection_list': lambda d: d['table']['rejected_for_no_joint_point'].pop(),
        'wrong_original_grid': lambda d: d['original']['grid'].__setitem__(0, .123),
        'selected_threshold': lambda d: d['table'].__setitem__('selected_threshold', .9),
        'wrong_summary_binding': lambda d: d['audit'].__setitem__('summary_sha256', '0' * 64),
        'wrong_record_binding': lambda d: d['audit']['models'][0].__setitem__('records_sha256', '0' * 64),
    }
    rejected = []
    for name, change in mutations.items():
        data = copy.deepcopy(fixture)
        change(data)
        try:
            checker.audit_loaded('frozen-encoder', **data)
        except (AssertionError, KeyError, StopIteration):
            rejected.append(name)
        else:
            raise AssertionError('Accepted corruption: ' + name)
    return dict(schema='mac-paired-diagnostic-table-auditor-review-v1', gpu_commit=commit,
        source_sha256=checker.sha(Path(__file__).read_bytes()),
        checker_sha256=checker.sha(Path(checker.__file__).read_bytes()), input_sha256=inputs,
        actual_old_points_checked=552, actual_old_group_dimensions_per_point=6,
        future_profile_labelled_fixture_points=552, corruption_rejections=rejected,
        original_raw_recomputed_summary=baseline['recomputed_summary'],
        direct_poisson_inversion_checks=review_poisson(),
        all_passed=True, new_frozen_model_results_checked=False,
        models_invoked=0, audio_accessed=False, candidate_qualified=False,
        limits='Validates the table auditor using authenticated old data and explicitly renamed parser fixtures. '
               'No new frozen-model recognition, private trace oracle, candidate quality or acceptance claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.gpu_ref)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('actual_old_points_checked', 'future_profile_labelled_fixture_points', 'corruption_rejections', 'all_passed')}))


if __name__ == '__main__':
    main()
