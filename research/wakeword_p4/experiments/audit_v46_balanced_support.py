"""Authenticate all saved balanced-training traces and recompute fit counts."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword/v46-balanced-training-support-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((base/'report.json').read_text());assert sha(base/'scores.npz')==r['scores_sha256']
counts={0:{},1:{}}
with np.load(base/'scores.npz') as z:
 assert len(z.files)==24192==len(r['details'])
 for d in r['details']:
  q=z[f"int8_{d['index']}"];m=d['modes']['int8'];assert hashlib.sha256(q.tobytes()).hexdigest()==m['score_sha256']
  five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');assert float(five.max())==m['full_max_five']
  for t,v in m['thresholds'].items():
   high=bool(five.max()>=float(t));assert high==v['full_high'];counts[d['label']][t]=counts[d['label']].get(t,0)+int(high)
for label,c in counts.items():
 for t,n in c.items():assert n==r['groups'][f'all/{label}']['modes']['int8/'+t]['full_high']
out=dict(traces=24192,all_hashes_match=True,full_high_counts=counts,report_sha256=sha(base/'report.json'),scores_sha256=sha(base/'scores.npz'),script_sha256=sha(Path(__file__)),limits='Saved hashes/maxima/counts only; no independent inference or container support replay.')
Path('research/wakeword_p4/results/4090-v46-balanced-training-support-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
