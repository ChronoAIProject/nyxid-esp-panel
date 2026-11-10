"""Attribute validation confusing-negative events to phrases, without final fixtures."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train_embedding_head import head, event_indices

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
report = json.loads((a.model/'training.json').read_text())
state = (a.model/'best.pt').read_bytes()
assert hashlib.sha256(state).hexdigest() == report['model_sha256']
assert hashlib.sha256((a.cache/'index.json').read_bytes()).hexdigest() == report['cache_index_sha256']
rows = json.loads((a.cache/'index.json').read_text())
raw = np.memmap(a.cache/'embeddings.bin', dtype='<f4', mode='r').reshape(-1, 96)
m = head(report.get('head_architecture', 'original')).cuda()
m.load_state_dict(torch.load(a.model/'best.pt', weights_only=True)); m.eval()
torch.set_num_threads(4)
threshold = report['selected_validation']['threshold']
groups = defaultdict(lambda: {'variants': 0, 'triggered': 0, 'events': 0, 'peak_scores': []})
for row in rows:
    if row['split'] != 'validation' or row['kind'] != 'generated' or row['label']:
        continue
    x = raw[row['offset']:row['offset']+row['frames']]
    windows = np.lib.stride_tricks.sliding_window_view(x, 16, axis=0).transpose(0, 2, 1)
    with torch.inference_mode():
        scores = m(torch.tensor(np.array(windows), device='cuda')).sigmoid().flatten().cpu().numpy()
    smoothed = np.convolve(scores, [.5, .5], mode='valid')
    events = event_indices(smoothed, threshold)
    g = groups[(row['engine'], row['text'])]
    g['variants'] += 1; g['triggered'] += bool(events); g['events'] += len(events)
    g['peak_scores'].append(float(max(smoothed)))
summary = []
for (engine, text), g in groups.items():
    values = g.pop('peak_scores')
    summary.append(dict(engine=engine, text=text, **g, trigger_rate=g['triggered']/g['variants'],
                        peak_median=float(np.median(values)), peak_max=max(values)))
summary.sort(key=lambda r: (-r['triggered'], -r['trigger_rate'], -r['peak_max']))
assert sum(x['variants'] for x in summary) == 3520
actual = sum(x['triggered'] for x in summary)/3520
assert abs(actual-report['selected_validation']['hard_negative_false_rate']) < 1e-12
result = dict(role='Validation-only phrase attribution; correlated synthetic rendering counts',
              model_sha256=report['model_sha256'], cache_index_sha256=report['cache_index_sha256'],
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), threshold=threshold,
              reproduced_hard_negative_false_rate=actual, regression_or_final_scored=False, phrases=summary)
a.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(summary[:12], indent=2))
