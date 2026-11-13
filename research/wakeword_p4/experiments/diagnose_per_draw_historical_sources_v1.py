"""Paired source diagnostics from consumed saved scores; no inference or tuning."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from evaluate import event_indices

BASE = Path('.local/wakeword')
OUT = Path('research/wakeword_p4/results/4090-per-draw-noise-historical-source-diagnostic-v1.json')
THRESHOLD = 0.98828125  # Already selected by original fixed historical calibration.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    groups = defaultdict(list)
    inputs = {}
    for corpus in ['phone', 'vctk']:
        roots = [BASE / f'per-draw-noise-{arm}-formal-v1-{corpus}-development-historical-threshold-v1'
                 for arm in ['fixed', 'per_draw']]
        reports = [json.loads((root / 'report.json').read_text()) for root in roots]
        for arm, root in zip(['fixed', 'per_draw'], roots):
            inputs[f'{arm}/{corpus}'] = sha(root / 'report.json')
        for i, (left, right) in enumerate(zip(reports[0]['records'], reports[1]['records'], strict=True)):
            assert left['record'] == right['record']
            row = left['record']
            assert row['split'] == 'validation' and row['role'] == 'development-paired-validation'
            measurements = []
            for root, item in zip(roots, [left, right]):
                path = root / f'{i}-scores.npy'
                assert sha(path) == item['nyx']['score_sha256']
                scores = np.load(path, allow_pickle=False)
                assert scores.ndim == 1 and np.isfinite(scores).all()
                times = np.asarray([(3 * (j + 4) + 5) * .01 for j in event_indices(scores, THRESHOLD)])
                hit = bool(np.any((times >= row['lead_samples'] / 16000)
                                  & (times <= row['keyword_end'] / 16000 + .8)))
                measurements.append(dict(max_score=float(scores.max()),
                                         trigger=hit if row['target'] == 'hey_nyx' else bool(len(times))))
            groups[(corpus, row['intended_phrase'])].append(dict(
                voice=row['voice'], path=row['path'], pcm_sha256=row['pcm_sha256'],
                fixed=measurements[0], per_draw=measurements[1]))
    summary = {}
    for (corpus, phrase), rows in groups.items():
        summary[f'{corpus}/{phrase}'] = dict(
            clips=len(rows), fixed_triggers=sum(r['fixed']['trigger'] for r in rows),
            per_draw_triggers=sum(r['per_draw']['trigger'] for r in rows),
            fixed_only=[r['path'] for r in rows if r['fixed']['trigger'] and not r['per_draw']['trigger']],
            per_draw_only=[r['path'] for r in rows if r['per_draw']['trigger'] and not r['fixed']['trigger']],
            median_max_score_delta=float(np.median([r['per_draw']['max_score'] - r['fixed']['max_score'] for r in rows])),
            records=rows)
    assert sum(g['clips'] for g in summary.values()) == 380
    result = dict(script_sha256=sha(Path(__file__)), input_report_sha256=inputs,
                  common_historical_threshold=THRESHOLD, groups=summary,
                  new_inference=False, threshold_tuned=False, candidate_promoted=False,
                  limits='Consumed historical source diagnostics only. Trigger interval uses full generation '
                  'boundary plus 800ms, not human phonetic truth. Source groups are correlated; this '
                  'diagnostic cannot establish independent generalization or a unique cause of failure.')
    with OUT.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'records'} for k, v in summary.items()}))


if __name__ == '__main__':
    main()
