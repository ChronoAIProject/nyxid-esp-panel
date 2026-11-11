"""Recount a frozen short screen from saved scores and original feature rows.

This is an artifact/accounting audit, not a second training or inference run.
Only training draws and the frozen old validation selection are replayed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'features', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    study, features = args.study, args.features
    result = json.loads((study/'result.json').read_bytes())
    plan_bytes = (study/'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    assert not result['smoke'] and not result['candidate_accepted_or_promoted']
    assert sha(plan_bytes) == result['plan_sha256']
    source_hashes = {}
    for name, key in (('screen_five_score_training.py', 'source_sha256'),
                      ('sequence_scorer.py', 'sequence_scorer_sha256')):
        source_hashes[name] = sha((Path(__file__).parent/name).read_bytes())
        assert source_hashes[name] == result[key]
    assert (repo/'research/wakeword_p4/training/plans/mac-five-score-short-screen-v1.json').read_bytes() == plan_bytes
    for name, expected in plan['feature_metadata_sha256'].items():
        assert sha((features/name).read_bytes()) == expected
    for name, key in (('best.weights.h5', 'initial_weights_sha256'),
                      ('training.json', 'initial_training_sha256')):
        assert sha((args.initial/name).read_bytes()) == plan[key]

    rows = json.loads((features/'features.json').read_bytes())
    selected = json.loads((study/'validation-selection.json').read_bytes())
    expected_selection, selected_indices = [], []
    for engine, label in sorted({(r['engine'], r['label']) for r in rows if r['split'] == 'validation'}):
        group = [(i, r) for i, r in enumerate(rows)
                 if r['split'] == 'validation' and (r['engine'], r['label']) == (engine, label)]
        hashes = sorted({r['sha256'] for _, r in group})[:16]
        chosen = [(i, r) for i, r in group if r['sha256'] in hashes and r['variant'] in (0, 3)]
        selected_indices.extend(i for i, _ in chosen)
        expected_selection.extend({'index': i, **r} for i, r in chosen)
    assert selected == expected_selection and len(selected) == result['validation_selected_rows'] == 334
    # The original selection artifact merges source metadata after `index`.
    # Some old source rows already have that key. Preserve the original bytes
    # and reconstruct actual feature row positions from the frozen rule.
    shadowed = [{'feature_row_index': i, 'stored_index': row['index'],
                 'path': row['path'], 'variant': row['variant']}
                for i, row in zip(selected_indices, selected, strict=True) if i != row['index']]
    counts = {}
    for engine in sorted({r['engine'] for r in selected}):
        counts[engine] = {}
        for label in (0, 1):
            group = [r for r in selected if r['engine'] == engine and r['label'] == label]
            counts[engine][str(label)] = {'feature_rows': len(group),
                'unique_source_sha256': len({r['sha256'] for r in group}),
                'unique_source_paths': len({r['path'] for r in group})}

    raw = np.memmap(features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    ambient_rows = {name: json.loads((features/name).read_bytes())
                    for name in ('human.json', 'ambient.json')}
    ambient_arrays = {}
    batch_hashes, draw_hashes, draw_splits, weights_hashes = {}, {}, {}, {}
    for arm in ('terminal', 'five_mean'):
        draw_bytes = (study/(arm+'-draws.jsonl')).read_bytes()
        draw_hashes[arm] = sha(draw_bytes)
        assert draw_hashes[arm] == result['arms'][arm]['draws_sha256']
        records = [json.loads(line) for line in draw_bytes.splitlines()]
        assert len(records) == result['arms'][arm]['steps'] == plan['steps'] == 500
        digest, splits = hashlib.sha256(), Counter()
        for step, record in enumerate(records, 1):
            assert record['step'] == step and len(record['draws']) == 16
            crops, labels = [], []
            for position, (name, index, start, end) in enumerate(record['draws']):
                label = int(position < 4)
                assert end-start == plan['sequence_crop_frames'] == 347 and start >= 0
                if position < 10:
                    assert name == 'features'
                    row = rows[index]
                    assert row['label'] == label and row['split'] == 'training'
                    assert end <= row['frames']
                    if label:
                        assert 15 <= end-row['keyword_frame'] <= 27
                    crop = raw[row['offset']+start:row['offset']+end]
                else:
                    row = ambient_rows[name][index]
                    assert row['split'] == 'training'
                    key = (name, index)
                    if key not in ambient_arrays:
                        ambient_arrays[key] = np.load(features/row['path'], mmap_mode='r', allow_pickle=False)
                    assert end <= len(ambient_arrays[key])
                    crop = ambient_arrays[key][start:end]
                assert crop.shape == (347, 40) and crop.dtype == np.uint16
                splits[row['split']] += 1
                crops.append(crop); labels.append(label)
            digest.update(np.asarray(crops, dtype='<u2').tobytes())
            digest.update(np.asarray(labels, dtype='<f4').tobytes())
        batch_hashes[arm] = digest.hexdigest()
        assert batch_hashes[arm] == result['arms'][arm]['batch_bytes_sha256']
        draw_splits[arm] = dict(splits)
        weights_hashes[arm] = sha((study/(arm+'.weights.h5')).read_bytes())
        assert weights_hashes[arm] == result['arms'][arm]['weights_sha256']
    assert batch_hashes['terminal'] == batch_hashes['five_mean']
    assert draw_hashes['terminal'] == draw_hashes['five_mean']

    summary, errors, score_counts, input_hashes = {}, {}, {}, []
    for arm in ('initial', 'terminal', 'five_mean'):
        evaluation = json.loads((study/(arm+'-evaluation.json')).read_bytes())
        assert evaluation == (result['initial_evaluation'] if arm == 'initial'
                              else result['arms'][arm]['evaluation'])
        assert sha((study/(arm+'-scores.npz')).read_bytes()) == evaluation['scores_sha256']
        groups, threshold_margin, inputs, total = {}, 1., [], 0
        with np.load(study/(arm+'-scores.npz'), allow_pickle=False) as arrays:
            assert set(arrays.files) == {'source_'+str(i) for i in selected_indices}
            for index, row, detail in zip(selected_indices, selected, evaluation['details'], strict=True):
                assert index == detail['index']
                for key in ('engine', 'label', 'path', 'sha256', 'variant'):
                    assert row[key] == detail[key]
                data = raw[row['offset']:row['offset']+row['frames']]
                feature_hash = sha(data.astype('<u2').tobytes())
                assert feature_hash == detail['input_feature_sha256']
                inputs.append((index, feature_hash))
                scores = arrays['source_'+str(index)]
                assert scores.ndim == 1 and scores.dtype == np.float32 and np.isfinite(scores).all()
                assert np.all((scores >= 0) & (scores <= 1))
                assert len(scores) == detail['scores'] == (len(data)-1-335)//3+1
                assert sha(scores.tobytes()) == detail['score_sha256']
                total += len(scores)
                armed, last, events = True, -1000, []
                # Independent scalar summation, mature score step112 onward.
                for at in range(4, len(scores)):
                    mean = math.fsum(float(v) for v in scores[at-4:at+1])/5
                    step = 112+at
                    threshold_margin = min(threshold_margin, abs(mean-plan['threshold']))
                    if mean < plan['threshold']/2 and step-last >= 67:
                        armed = True
                    if armed and mean >= plan['threshold'] and step-last >= 67:
                        events.append(step); armed, last = False, step
                offsets = [480*(step+1) for step in events]
                assert events == detail['mature_event_steps'] and offsets == detail['return_offsets']
                assert bool(events) == detail['any_event']
                end = (row['keyword_frame']+2)*160 if row['label'] else None
                hit = any(end-1600 <= sample <= end+12800 for sample in offsets) if end is not None else None
                assert hit == detail['estimated_end_window_hit']
                group = groups.setdefault(row['engine'], Counter({
                    'positive_clips': 0, 'negative_clips': 0, 'positive_any_hits': 0,
                    'positive_estimated_window_hits': 0, 'negative_false_clips': 0}))
                group['positive_clips' if row['label'] else 'negative_clips'] += 1
                if row['label']:
                    group['positive_any_hits'] += bool(events)
                    group['positive_estimated_window_hits'] += hit
                else:
                    group['negative_false_clips'] += bool(events)
        groups = {key: dict(value) for key, value in groups.items()}
        assert groups == evaluation['summary']
        summary[arm] = {'by_engine': groups,
            'total': dict(sum((Counter(value) for value in groups.values()), Counter()))}
        errors[arm] = threshold_margin
        score_counts[arm] = total
        input_hashes.append(inputs)
    assert input_hashes[0] == input_hashes[1] == input_hashes[2]
    report = {'scope': __doc__, 'audit_source_sha256': sha(Path(__file__).read_bytes()),
        'study_result_sha256': sha((study/'result.json').read_bytes()),
        'plan_sha256': sha(plan_bytes), 'training_source_hashes': source_hashes,
        'source_counts': counts, 'validation_rows': len(selected),
        'actual_validation_feature_row_indices': selected_indices,
        'selection_artifact_shadowed_index_rows': shadowed,
        'batch_bytes_sha256': batch_hashes, 'draws_sha256': draw_hashes,
        'replayed_draw_splits': draw_splits, 'weights_sha256': weights_hashes,
        'scores_checked': score_counts, 'smallest_threshold_distance': errors,
        'all_evaluation_input_hashes_match': True, 'summary': summary,
        'candidate_promoted': False,
        'limits': ['Not independent training, inference or verification of process filesystem access.',
            'Validation source hashes/paths and feature variants are correlated; counts are not independent trials.',
            'Old synthetic validation, fixed threshold, one seed; no acoustic, background FAR, INT8 or P4 acceptance.']}
    with args.output.open('x') as out:
        out.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('validation_rows', 'source_counts', 'summary', 'scores_checked')}))


if __name__ == '__main__':
    main()
