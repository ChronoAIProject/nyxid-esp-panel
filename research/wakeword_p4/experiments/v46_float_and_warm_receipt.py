"""Saved-array audit: v46 development float and warm-window training counts."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import event_indices,smooth
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=base/'v46-old-development-float-probe-v1';r=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==r['scores_sha256'];decisions=collections.Counter()
with np.load(root/'scores.npz') as z:
 assert len(z.files)==380
 for d in r['details']:
  f=z[f"{d['protocol']}_{d['index']}"];assert float(f.max())==d['float_max_five'];assert event_indices(f,r['threshold'])==d['float_events']
  if d['record']['target']!='hey_nyx':
   a,b=bool(d['float_events']),bool(d['int8_events']);decisions['both' if a and b else 'float_only' if a else 'int8_only' if b else 'neither']+=1
warm={}
for v in ['v45','v46']:
 root=base/f'{v}-balanced-training-support-v1';report=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==report['scores_sha256'];groups={}
 with np.load(root/'scores.npz') as z:
  for d in report['details']:
   q=z[f"int8_{d['index']}"];assert hashlib.sha256(q.tobytes()).hexdigest()==d['modes']['int8']['score_sha256']
   for t in [.5,.992,.99609375]:
    g=groups.setdefault(f"{d['label']}/{t}",dict(variants=0,full_trace_event_clips=0,warmed_five_high_clips=0,warmed_restarted_event_clips=0,container_support_hit_clips=0));g['variants']+=1
    full=event_indices(smooth(q,5),t);w=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');old=d['modes']['int8']['thresholds'][str(t)];assert full==old['event_indices']
    g['full_trace_event_clips']+=bool(full);g['warmed_five_high_clips']+=bool(np.any(w>=t));g['warmed_restarted_event_clips']+=bool(event_indices(w,t));g['container_support_hit_clips']+=old['support_high_positions']>0
 warm[v]=groups
out=dict(development_negative_decisions=dict(decisions),development_float_arrays=380,training_arrays=48384,warm_counts=warm,script_sha256=sha(Path(__file__)),limits='Warmed restarted event detector begins at first complete five after q[49:]; differs from full-trace latch history. Saved arrays only; no inference or new data. Container support is a process ledger, not phonetic truth.')
Path('research/wakeword_p4/results/4090-v46-float-and-warm-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
