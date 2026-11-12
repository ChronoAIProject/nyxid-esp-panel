"""Authenticate consumed development traces and describe fixed-grid failure only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import event_indices

BASE = Path('.local/wakeword')
ROOT = BASE/'matched-noise-independent-formal-v1-slr83-development-v1'
CAL = BASE/'matched-noise-independent-formal-v1-slr83-new-development-calibration-v1.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads((ROOT/'report.json').read_text())
c = json.loads(CAL.read_text())
assert r['model_sha256'] == c['model_sha256']
assert c['candidate_report_sha256'] == sha(ROOT/'report.json')
assert c['threshold'] is None and not c['viable_operating_point']
assert len(c['grid']) == 138
rows = []
for i, item in enumerate(r['records']):
    row = item['record']
    path = ROOT/f'{i}-scores.npy'
    assert sha(path) == item['nyx']['score_sha256']
    scores = np.load(path, allow_pickle=False)
    assert scores.ndim == 1 and np.isfinite(scores).all()
    if row['target'] == 'hey_nyx':
        continue
    counts = [len(event_indices(scores, p['threshold'])) for p in c['grid']]
    if not any(counts):
        continue
    rows.append(dict(path=row['path'], voice=row['voice'], phrase=row['intended_phrase'],
        speed=row['length_multiplier'], variant=row['variant'],
        pcm_sha256=row['pcm_sha256'], score_sha256=sha(path),
        max_smoothed_score=float(scores.max()),
        fixed_grid_points_with_events=sum(x > 0 for x in counts),
        events_at_grid_max=counts[-1]))
assert sum(x['events_at_grid_max'] > 0 for x in rows) == sum(
    g['false_activation_clips'] for g in c['grid'][-1]['synthetic']['negatives'].values())
out = dict(model_sha256=r['model_sha256'], calibration_sha256=sha(CAL),
    synthetic_report_sha256=sha(ROOT/'report.json'), script_sha256=sha(Path(__file__)),
    threshold_grid_unchanged=True, rows=rows,
    limits='Consumed development trace diagnosis only. No new inference, threshold extension, '
           'retraining, source regeneration, final access or reversal of candidate rejection. '
           'A confident score does not certify acoustic phonetic truth or causal mechanism.')
p = Path('research/wakeword_p4/results/4090-matched-noise-reserved-failure-audit-v1.json')
assert not p.exists()
p.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(rows, indent=2))
