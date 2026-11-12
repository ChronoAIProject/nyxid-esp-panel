"""Inventory old trace failures at every original grid point; no inference/tuning."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from evaluate import event_indices

BASE = Path('research/wakeword_p4/results')
LOCAL = Path('.local/wakeword')
out = {'arms': {}, 'limits': 'Reused historical diagnostics only. All original thresholds retained; no new '
       'operating point or independent acceptance. Voices and renders correlated. Generation boundaries '
       'are synthetic references, not phoneme truth. Background recording IDs are not person identities.'}
for arm in ('control', 'treatment'):
    prefix = f'temporal-tail-risk-{arm}-formal-v1'
    calpath = BASE / f'4090-{prefix}-reused-development-exploratory-calibration-v1.json'
    cal = json.loads(calpath.read_bytes())
    grids = cal['grid']
    armout = {'input_sha256': {str(calpath): hashlib.sha256(calpath.read_bytes()).hexdigest()},
              'model_sha256': cal['model_sha256'], 'synthetic': [], 'background': []}
    for corpus in ('phone', 'vctk'):
        tag = f'{prefix}-{corpus}-development-historical-threshold-v1'
        reportpath = BASE / f'4090-{tag}-report.json'
        payload = reportpath.read_bytes()
        assert (LOCAL / tag / 'report.json').read_bytes() == payload
        report = json.loads(payload)
        assert report['model_sha256'] == cal['model_sha256'] and not report['final_data_used']
        armout['input_sha256'][str(reportpath)] = hashlib.sha256(payload).hexdigest()
        for i, item in enumerate(report['records']):
            path = LOCAL / tag / f'{i}-scores.npy'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item['nyx']['score_sha256']
            scores = np.load(path, allow_pickle=False)
            assert scores.ndim == 1 and len(scores) and np.isfinite(scores).all()
            record = item['record']
            assert record['split'] == 'validation' and record['role'] == 'development-paired-validation'
            row = {'corpus': corpus, 'record': record, 'score_sha256': item['nyx']['score_sha256'], 'grid': []}
            for point in grids:
                indices = event_indices(scores, point['threshold'])
                times = np.asarray([(3 * (index + 4) + 5) * .01 for index in indices])
                entry = {'threshold': point['threshold'], 'events': len(indices)}
                if record['target'] == 'hey_nyx':
                    entry['presentation_trigger'] = bool(np.any((times >= record['lead_samples']/16000)
                                                               & (times <= record['keyword_end']/16000 + .8)))
                    entry['pre_generation_events'] = int(np.count_nonzero(times < record['lead_samples']/16000))
                row['grid'].append(entry)
            armout['synthetic'].append(row)
    tag = f'{prefix}-voxconverse-development-v1'
    reportpath = BASE / f'4090-{tag}-report.json'
    payload = reportpath.read_bytes()
    assert (LOCAL / tag / 'report.json').read_bytes() == payload
    report = json.loads(payload)
    assert report['model_sha256'] == cal['model_sha256'] and not report['independent_final_scored']
    armout['input_sha256'][str(reportpath)] = hashlib.sha256(payload).hexdigest()
    for item in report['speakers']:
        path = LOCAL / tag / (item['speaker'] + '-scores.npy')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['score_sha256']
        scores = np.load(path, allow_pickle=False)
        assert scores.ndim == 1 and len(scores) and np.isfinite(scores).all()
        armout['background'].append({
            'recording': item['speaker'], 'source_seconds': item['source_seconds'],
            'score_sha256': item['score_sha256'],
            'events_by_grid': [len(event_indices(scores, point['threshold'])) for point in grids]})
    # Reconstruct every original aggregate, making the inventory accountable to the gate.
    for j, point in enumerate(grids):
        recalls = []
        for corpus in ('phone', 'vctk'):
            rows = [r for r in armout['synthetic'] if r['corpus'] == corpus]
            positives = [r for r in rows if r['record']['target'] == 'hey_nyx']
            group = next(g for g in point['groups'].values() if g['positives'] == len(positives))
            hits = sum(r['grid'][j]['presentation_trigger'] for r in positives)
            assert hits == group['triggered']
            assert sum(r['grid'][j]['pre_generation_events'] for r in positives) == group['pre_generation_positive_events']
            assert sum(r['grid'][j]['events'] > 0 for r in rows if r['record']['target'] != 'hey_nyx') == group['false_activation_clips']
            recalls.append(hits / len(positives))
        assert float(np.mean(recalls)) == point['synthetic_macro_clip_trigger']
        assert sum(r['events_by_grid'][j] for r in armout['background']) == point['background_events']
    print(arm, 'authenticated traces', len(armout['synthetic']) + len(armout['background']))
    print('Residual negatives at original maximum threshold:', [r['record']['path'] for r in armout['synthetic']
          if r['record']['target'] != 'hey_nyx' and r['grid'][-1]['events']])
    armout['thresholds'] = [point['threshold'] for point in grids]
    for row in armout['synthetic']:
        entries = row.pop('grid')
        row['events_by_grid'] = [entry['events'] for entry in entries]
        if row['record']['target'] == 'hey_nyx':
            row['presentation_trigger_by_grid'] = [entry['presentation_trigger'] for entry in entries]
            row['pre_generation_events_by_grid'] = [entry['pre_generation_events'] for entry in entries]
    out['arms'][arm] = armout
(BASE / '4090-temporal-tail-risk-failure-inventory-v1.json').write_text(json.dumps(out, separators=(',', ':')) + '\n')
