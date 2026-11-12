"""Audit existing continuation scores without inference or new threshold search.

Post-hoc descriptive analysis of the original nineteen-point validation grid.
It corrects an overstatement about strict eligibility, and compares the two
independently selected checkpoints at the original ensemble threshold .4.
No new examples, trained weights or deployment decisions are produced.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event_count(scores, threshold):
    count, ready, since = 0, True, 1000
    for value in scores:
        if value < threshold / 2 and since >= 25:
            ready = True
        if ready and value >= threshold and since >= 25:
            count += 1
            ready, since = False, 0
        since += 1
    return count


def identity(row):
    return {k: v for k, v in row.items() if k not in ('scores', 'max_score', 'events')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--continuation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and __debug__
    summary_path = args.continuation / 'result-summary.json'
    assert sha(summary_path) == '24edcc591c0460e93cf65c43d3d9b24bb499b89c473d714b517103d6ba509fd2'
    summary = json.loads(summary_path.read_bytes())
    result = dict(scope=__doc__, archived_original_summary_sha256=sha(summary_path),
        summary_location_in_archive='run/result-summary.json', script_sha256=sha(Path(__file__)),
        fixed_comparison_threshold=.4, arms={}, paired_groups={}, accepted=False,
        correction='The preselected operating points fail strict criteria, but other original grid points do satisfy the low-false criteria at much lower recall. It was incorrect to say no strict eligible point exists. At an identical threshold the treatment can lower false positives while also losing recall; independently selected-threshold metrics do not prove uniformly worse separation.',
        limits='Post-hoc descriptive audit of consumed validation only, with independently selected dense checkpoints7500/5000 and a shared frozen temporal member. No unique causal attribution, new threshold selection, inference, optimization, unused-development/final, audio or board access. Original candidate rejection remains in force.')
    all_clips = {}
    for arm in ('control', 'treatment'):
        recorded = summary['arms'][arm]
        plan_path = HERE / 'plans' / ('us-embedding-mean-' + arm + '-v1.json')
        assert sha(plan_path) == recorded['mean_plan_sha256']
        plan = json.loads(plan_path.read_bytes())
        calibration = args.continuation / ('mean-' + arm) / 'calibration.json'
        details = calibration.with_name('details-private.json')
        assert sha(calibration) == recorded['mean_calibration_sha256']
        assert sha(details) == recorded['mean_details_sha256']
        saved, mean = json.loads(calibration.read_bytes()), json.loads(details.read_bytes())
        members, roots = [], []
        for member in plan['members']:
            root = args.models / member['checkpoint'] / 'validation'
            assert sha(root / 'details-private.json') == member['details_sha256']
            assert {p.name: sha(p) for p in root.glob('*-scores.npy')} == member['continuous_scores']
            members.append(json.loads((root / 'details-private.json').read_bytes()))
            roots.append(root)
        assert [len(m['clips']) for m in members] == [len(mean['clips'])] * 2
        assert [len(m['continuous']) for m in members] == [len(mean['continuous'])] * 2
        for left, right, actual in zip(members[0]['clips'], members[1]['clips'], mean['clips']):
            assert identity(left) == identity(right) == identity(actual)
            a, b = np.asarray(left['scores']), np.asarray(right['scores'])
            assert a.shape == b.shape and np.max(np.abs(a * 768 - np.rint(a * 768))) < 1e-9
            assert np.max(np.abs(b * 768 - np.rint(b * 768))) < 1e-9
            exact = (np.rint(a * 768).astype(np.int64) + np.rint(b * 768).astype(np.int64)) / 1536
            np.testing.assert_array_equal(exact, actual['scores'])
            valid = [(not actual['label']) or actual['keyword_end'] - .1 <= t <= actual['keyword_end'] + .8
                     for t in actual['times']]
            expected_max = max((s for s, use in zip(exact, valid) if use), default=0.)
            assert actual['max_score'] == expected_max
        thresholds = plan['event_policy']['threshold_grid']
        assert [r['threshold'] for r in saved['grid']] == thresholds
        for left, right, actual in zip(members[0]['continuous'], members[1]['continuous'], mean['continuous']):
            assert identity(left) == identity(right) == identity(actual)
            name = actual['id'] + '-scores.npy'
            a, b = [np.load(root / name, allow_pickle=False) for root in roots]
            assert a.shape == b.shape
            assert np.max(np.abs(a * 768 - np.rint(a * 768))) < 1e-9
            assert np.max(np.abs(b * 768 - np.rint(b * 768))) < 1e-9
            exact = (np.rint(a * 768).astype(np.int64) + np.rint(b * 768).astype(np.int64)) / 1536
            assert len(exact) * .08 / 3600 == actual['evaluated_hours']
            for threshold in thresholds:
                assert event_count(exact, threshold) == actual['events'][str(threshold)]
        for point in saved['grid']:
            threshold = point['threshold']
            positive, negative = defaultdict(list), defaultdict(list)
            for row in mean['clips']:
                key = row['engine'] if row['label'] else row['engine'] + ':' + row['corpus']
                (positive if row['label'] else negative)[key].append(row['max_score'] >= threshold)
            recall = {k: sum(v) / len(v) for k, v in positive.items()}
            assert recall == point['recall_by_engine']
            assert abs(sum(recall.values()) / len(recall) - point['macro_recall']) < 1e-12
            for k, values in negative.items():
                assert point['confusing_corpora'][k] == dict(variants=len(values), false_variants=sum(values), false_rate=sum(values) / len(values))
            assert sum(r['events'][str(threshold)] for r in mean['continuous']) == point['continuous_events']
        strict = [r for r in saved['grid'] if r['continuous_events_per_hour'] <= .5 and r['max_corpus_false_rate'] <= .01]
        assert strict and not saved['strict_operating_point']
        result['arms'][arm] = dict(mean_calibration_sha256=sha(calibration), mean_details_sha256=sha(details),
            verified_clip_means=len(mean['clips']), verified_continuous_means=len(mean['continuous']),
            original_grid_points_verified=len(thresholds), original_low_false_points=strict,
            highest_macro_recall_with_original_low_false_criteria=max(r['macro_recall'] for r in strict),
            same_threshold_point=next(r for r in saved['grid'] if r['threshold'] == .4),
            independently_selected_point=saved['selected'])
        all_clips[arm] = mean['clips']
    assert len(all_clips['control']) == len(all_clips['treatment'])
    paired = defaultdict(Counter)
    for left, right in zip(all_clips['control'], all_clips['treatment']):
        assert identity(left) == identity(right)
        group = ('positive/' if left['label'] else 'negative/') + left['engine'] + ':' + left['corpus']
        state = ('high' if left['max_score'] >= .4 else 'low') + '_to_' + ('high' if right['max_score'] >= .4 else 'low')
        paired[group][state] += 1
    result['paired_groups'] = {k: dict(v) for k, v in sorted(paired.items())}
    result['all_checks_pass'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('arms', 'paired_groups')}, indent=2))


if __name__ == '__main__':
    main()
