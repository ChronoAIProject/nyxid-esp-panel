"""Recompute saved training-only neighbours with SciPy correlation costs.

This audits feature geometry, not pronunciation or classifier acceptance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference(left, right):
    arrays = [a.astype(np.float64)/25.6 for a in (left, right)]
    live = [np.linalg.norm(a-a.mean(1, keepdims=True), axis=1) > 1e-12 for a in arrays]
    cost = np.ones((len(left), len(right)), np.float64)
    cost[np.ix_(live[0], live[1])] = np.clip(cdist(arrays[0][live[0]], arrays[1][live[1]], 'correlation'), 0, 2)
    cost[np.ix_(~live[0], ~live[1])] = 0
    table = np.full((len(left)+1, len(right)+1), np.inf); table[0, 0] = 0
    for i in range(len(left)):
        for j in range(len(right)):
            table[i+1, j+1] = cost[i, j]+min(table[i, j], table[i, j+1], table[i+1, j])
    return float(table[-1, -1]*2/(len(left)+len(right)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh audit output file')
    root = args.artifacts.resolve()
    read = lambda name: json.loads((root/name).read_bytes())
    plan, result, inventory, groups, comparisons, review = [read(n) for n in
        ('plan.json', 'result.json', 'inventory.json', 'groups.json', 'comparisons.json', 'review-candidates.json')]
    assert sha(root/'plan.json') == result['plan_sha256']
    assert sha(args.features/'features.json') == plan['feature_index_sha256']
    assert sha(args.features/'features.bin') == plan['feature_data_sha256']
    for name, expected in result['evidence_sha256'].items():
        assert sha(root/name) == expected, ('Evidence differs', name)
    for name, expected in plan['source_sha256'].items():
        assert sha(root/'source'/name) == expected
    index = json.loads((args.features/'features.json').read_bytes())
    lookup = {(r['path'], r['variant']): r for r in index}
    reserved = {r['sha256'] for r in index if r['split'] != 'training'}
    raw = np.memmap(args.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    cached = np.load(root/'features.npz', allow_pickle=False)
    assert set(cached.files) == {f'source_{i}' for i in range(len(inventory))}
    source_classes = defaultdict(set)
    for number, item in enumerate(inventory):
        row = lookup[item['path'], 0]
        assert row['split'] == 'training' and row['sha256'] not in reserved
        assert item['id'] == number and item['source_sha256'] == row['sha256']
        assert item['text'] == row['text'] and item['engine'] == row['engine']
        assert row['label'] == int(item['class'] == 0)
        source_classes[row['sha256']].add(item['class'])
        value = cached[f'source_{number}']; lo, hi = item['feature_bounds']
        assert value.dtype == np.uint16 and value.shape == (hi-lo, 40) and 0 <= lo < hi <= row['frames']
        np.testing.assert_array_equal(value, raw[row['offset']+lo:row['offset']+hi])
        assert hashlib.sha256(value.tobytes()).hexdigest() == item['feature_sha256']
    assert all(len(values) == 1 for values in source_classes.values()), 'Exact waveform/intent collision'
    matrices = np.load(root/'distances.npz', allow_pickle=False)
    assert set(matrices.files) == {f'group_{i}' for i in range(len(groups))}
    checked, maximum_delta, expected_comparisons = 0, 0., []
    for number, group in enumerate(groups):
        ids = group['source_ids']; matrix = matrices[f'group_{number}']
        positive = [i for i, uid in enumerate(ids) if inventory[uid]['class'] == 0]
        assert matrix.shape == (len(ids), len(ids)) and group['control_available'] == (len(positive) >= 2)
        assert len({inventory[i]['source_sha256'] for i in ids}) == len(ids)
        for uid, aliases in zip(ids, group['aliases'], strict=True):
            assert aliases[0] == uid
            assert all(inventory[i]['source_sha256'] == inventory[uid]['source_sha256'] for i in aliases)
            assert all(inventory[i]['voice_group'] == group['voice_group'] for i in aliases)
        for i in range(len(ids)):
            assert matrix[i, i] == 0
            for j in range(i+1, len(ids)):
                if i not in positive and j not in positive:
                    assert np.isnan(matrix[i, j]) and np.isnan(matrix[j, i]); continue
                expected = reference(cached[f'source_{ids[i]}'], cached[f'source_{ids[j]}'])
                delta = abs(expected-matrix[i, j]); maximum_delta = max(maximum_delta, delta)
                assert delta <= 1e-12 and matrix[i, j] == matrix[j, i]
                checked += 1
        if len(positive) < 2:
            continue
        for i in positive:
            j = min((j for j in positive if j != i), key=lambda j: (matrix[i, j], ids[j]))
            for label in (1, 2, 3):
                negative = [j for j, uid in enumerate(ids) if inventory[uid]['class'] == label]
                if negative:
                    k = min(negative, key=lambda j: (matrix[i, j], ids[j]))
                    expected_comparisons.append({'engine': inventory[ids[i]]['engine'], 'class': label,
                        'anchor_id': ids[i], 'positive_id': ids[j], 'negative_id': ids[k],
                        'positive_distance': float(matrix[i, j]), 'negative_distance': float(matrix[i, k]),
                        'margin': float(matrix[i, k]-matrix[i, j])})
    assert comparisons == expected_comparisons
    expected_review, cohorts = [], []
    for engine, label in sorted({(r['engine'], r['class']) for r in comparisons}):
        rows = [r for r in comparisons if (r['engine'], r['class']) == (engine, label)]
        margins = np.array([r['margin'] for r in rows])
        cohorts.append({'engine': engine, 'class': label, 'anchors': len(rows),
            'negative_closer_than_positive_reference': int(sum(margins < 0)), 'ties': int(sum(margins == 0)),
            'median_margin': float(np.median(margins))})
        expected_review.extend(sorted(rows, key=lambda r: (r['margin'], r['anchor_id'], r['negative_id']))[:plan['review_per_engine_class']])
    assert cohorts == result['cohorts'] and review == expected_review
    assert len(inventory) == result['source_paths'] == plan['expected_source_paths']
    assert len(source_classes) == result['source_hashes'] == plan['expected_source_hashes']
    assert checked == result['computed_pair_distances'] and result['accepted'] is False
    audit = {'result_sha256': sha(root/'result.json'), 'auditor_sha256': sha(Path(__file__)),
        'source_paths': len(inventory), 'checked_pair_distances': checked,
        'maximum_independent_distance_delta': maximum_delta, 'numeric_allowance': 1e-12,
        'all_features_match_original_cache': True, 'exact_source_intent_conflicts': 0,
        'all_comparisons_cohorts_and_review_selection_reconstructed': True,
        'accepted': False, 'numpy': np.__version__,
        'limits': 'Independent cached-geometry accounting; no phonetic verification, quality advancement or new model inference.'}
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
