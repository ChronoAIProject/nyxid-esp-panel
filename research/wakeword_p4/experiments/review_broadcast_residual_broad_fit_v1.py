"""Independently decode saved full broad-fit probabilities, without inference."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');p=R/'4090-broadcast-residual-broad-fit-original-v1.json';d=json.loads(p.read_bytes());t=d['trace'];raw=Path(t['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==t['sha256']
with np.load(t['path'],allow_pickle=False) as z:v=z['probability'].copy()
assert v.shape==(31644,282) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
groups={}
for r,a in zip(d['records'],v,strict=True):
 sums=np.convolve(a.astype(np.float64),np.ones(5),mode='valid');armed=True;last=-1000;events=[]
 for step,total in enumerate(sums,5):
  if step-last<67:continue
  if total<(235/256)*2.5:armed=True
  if armed and total>=(235/256)*5:events.append(step);last=step;armed=False
 assert events==r['events'],r['index']
 g=groups.setdefault(r['engine']+'/'+r['text'],dict(rows=0,positive_crossing_rows=0,valid_event_rows=0,padded_only_event_rows=0));g['rows']+=1;g['positive_crossing_rows']+=bool(r['positive_crossings']);g['valid_event_rows']+=any(s<=r['valid_calls'] for s in events);g['padded_only_event_rows']+=bool(events) and all(s>r['valid_calls'] for s in events)
assert groups==d['groups']
out=dict(all_event_checks_passed=True,rows_verified=31644,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),groups=groups,limits='Independent float event decoding and aggregate reconstruction. Positive-mask crossing lists are aggregated as reported, not independently recomputed. Training-only zero-cache float fit, not independent acceptance/P4 or pronunciation verification.')
with (R/'4090-broadcast-residual-broad-fit-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(groups))
