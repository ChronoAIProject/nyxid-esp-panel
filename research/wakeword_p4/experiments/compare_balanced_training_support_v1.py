"""Matched balanced-source INT8 census; training fit versus previous model transfer."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reports={};inputs={}
for version in ['v45','v46']:
 root=base/f'{version}-balanced-training-support-v1';r=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==r['scores_sha256'];assert len(r['details'])==24192
 with np.load(root/'scores.npz') as z:
  assert len(z.files)==24192
  for i,d in enumerate(r['details']):
   assert d['index']==i;q=z[f'int8_{i}'];m=d['modes']['int8'];assert hashlib.sha256(q.tobytes()).hexdigest()==m['score_sha256']
   five=np.convolve(q[49:].astype(np.float64),np.ones(5)/5,mode='valid');assert float(five.max())==m['full_max_five']
   for t,h in m['thresholds'].items():assert bool(five.max()>=float(t))==h['full_high']
 reports[version]=r;inputs[version]={'report_sha256':sha(root/'report.json'),'scores_sha256':r['scores_sha256'],'model_sha256':r['model_sha256']}
a,b=reports['v45'],reports['v46'];assert a['pack_sha256']==b['pack_sha256'];paired={}
for x,y in zip(a['details'],b['details']):
 for k in ['index','source','voice','target','variant','label','valid_support_positions']:assert x[k]==y[k]
 for t in ['0.5','0.992','0.99609375']:
  for metric in ['full_high','support_high_positions']:
   p=bool(x['modes']['int8']['thresholds'][t][metric]);q=bool(y['modes']['int8']['thresholds'][t][metric]);key=f"{x['label']}/{t}/{metric}"
   g=paired.setdefault(key,{'neither':0,'v45_only':0,'v46_only':0,'both':0});g['both' if p and q else 'v45_only' if p else 'v46_only' if q else 'neither']+=1
out={'scope':'All24192 identical balanced training variants; v45 transfer versus v46 training fit','inputs':inputs,'groups':{v:{k:r['groups'][k] for k in ['all/0','all/1']} for v,r in reports.items()},'paired':paired,'authenticated_traces':48384,'script_sha256':sha(Path(__file__)),'limits':'This pack trained v46 but not v45. Not independent generalization, isolated causality, validation, threshold selection or acceptance. Selected checkpoints and training trajectories differ. No new/final data used.'}
Path('research/wakeword_p4/results/4090-balanced-training-support-comparison-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(paired))
