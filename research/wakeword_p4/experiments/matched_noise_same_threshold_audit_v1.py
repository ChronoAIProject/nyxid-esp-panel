"""Describe the frozen historical grid; perform no inference or threshold selection."""
import hashlib
import json
from pathlib import Path

BASE = Path('research/wakeword_p4/results')
inputs = {arm: BASE / f'4090-matched-noise-{arm}-formal-v1-reused-development-exploratory-calibration-v1.json'
          for arm in ('independent', 'common')}
data = {arm: json.loads(path.read_bytes()) for arm, path in inputs.items()}
rows = []
for independent, common in zip(data['independent']['grid'], data['common']['grid'], strict=True):
    assert independent['threshold'] == common['threshold']
    assert independent['background_hours'] == common['background_hours']
    assert set(independent['groups']) == set(common['groups'])
    row = {'threshold': independent['threshold']}
    for group in independent['groups']:
        for key in ('positives', 'negative_clips'):
            assert independent['groups'][group][key] == common['groups'][group][key]
    for arm, point in (('independent', independent), ('common', common)):
        row[arm] = {
            'macro': point['synthetic_macro_clip_trigger'],
            'negative_false_clips': sum(g['false_activation_clips'] for g in point['groups'].values()),
            'pre_generation_positive_events': sum(g['pre_generation_positive_events'] for g in point['groups'].values()),
            'background_events': point['background_events'],
            'background_upper_95': point['background_upper_95'],
            'eligible': point['eligible'],
        }
    rows.append(row)
assert len(rows) == 138
counts = {}
for metric in ('macro', 'negative_false_clips', 'background_events'):
    counts[metric] = {
        relation: sum(compare(row['common'][metric], row['independent'][metric]) for row in rows)
        for relation, compare in (('lower', lambda a, b: a < b),
                                  ('equal', lambda a, b: a == b),
                                  ('higher', lambda a, b: a > b))
    }
counts['same_threshold_recall_gain_without_more_false_clips_or_background'] = sum(
    row['common']['macro'] > row['independent']['macro']
    and row['common']['negative_false_clips'] <= row['independent']['negative_false_clips']
    and row['common']['background_events'] <= row['independent']['background_events']
    for row in rows)
out = {
    'input_sha256': {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in inputs.items()},
    'model_sha256': {arm: report['model_sha256'] for arm, report in data.items()},
    'counts': counts, 'rows': rows,
    'limits': 'All 138 original historical points; correlated thresholds, single matched seed. '
              'No selected operating point, new inference, data access, coefficient tuning or acceptance. '
              'Differences do not establish a unique causal mechanism or independent generalization.',
}
(BASE / '4090-matched-noise-same-threshold-audit-v1.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(counts, indent=2))
