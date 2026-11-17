"""Independently recompute all saved factorial metrics; no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'schema':'multiscale-longhistory-review-v1','arms':{},'all_passed':False}
sources=None
for arm in ['control','multiscale']:
 plan=R/f'4090-multiscale-longhistory-{arm}-plan-v1.json';report=R/f'4090-multiscale-longhistory-{arm}-original-v1.json';p=json.loads(plan.read_text());s=json.loads(report.read_text());assert s['plan_sha256']==sha(plan) and s['optimizer_updates']==0
 for f,h in p['files'].items():assert sha(f)==h,f
 assert len(s['records'])==90 and [r['source'] for r in s['records']]==p['positives']
 if sources is None:sources=p['positives']
 else:assert sources==p['positives']
 assert sha(s['trace']['path'])==s['trace']['sha256'];summary={};failures=[];verified=0;expected=set()
 with np.load(s['trace']['path']) as z:
  for i,r in enumerate(s['records']):
   assert r['index']==i and r['split_feature']%3==0
   text=r['source']['text'];g=summary.setdefault(text,{})
   for f in ['silence','speech']:
    key=f'{i}_{f}_features';expected.add(key);x=z[key];assert x.ndim==2 and x.shape[1]==40 and np.isfinite(x).all()
    for c in ['silence','speech']:
     key=f'{i}_{f}_{c}_probability';expected.add(key);v=z[key];assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all();assert len(v)==(len(x)-r['split_feature'])//3
     avg=np.lib.stride_tricks.sliding_window_view(v.astype(np.float64),5).mean(axis=1)
     end=(np.arange(len(v))+1+r['split_feature']//3)*480+320
     prefix=(end[4:]>=r['prefix_samples'])&(end[4:]<=r['prefix_samples']+r['source']['prefix_reference_samples']+4320)
     whole=(end[4:]>=r['prefix_samples'])&(end[4:]<=r['prefix_samples']+r['source']['native_samples']+4320)
     d=r['arms'][f'{f}/{c}'];assert np.isclose(avg[prefix].max(),d['max_five'],atol=1e-12,rtol=0);assert np.isclose(avg[whole].max(),d['whole_container_max_five'],atol=1e-12,rtol=0)
     hit=bool((avg[prefix]>=.91796875).any());assert hit==d['crossing'];assert bool((avg[whole]>=.91796875).any())==d['whole_container_crossing']
     if f==c:assert d['diagonal_max_error']<=1e-6
     g[f'{f}/{c}']=g.get(f'{f}/{c}',0)+int(hit);verified+=1
     if not hit:failures.append({'index':i,'source_id':r['source']['id'],'text':text,'condition':f'{f}/{c}','max_five':d['max_five']})
  assert set(z.files)==expected
 assert verified==360
 out['arms'][arm]={'report_sha256':sha(report),'trace_sha256':s['trace']['sha256'],'conditions_verified':verified,'summary':summary,'failures':failures}
out['all_passed']=True;out['reviewer_sha256']=sha(__file__);out['limits']='Training-native source attribution only; reused fixed90sources, artificial offdiagonal cache transfers. Saved-array metric review does not independently rerun full/split inference. No P4/acceptance/generalization claim.'
(R/'4090-multiscale-longhistory-review-v1.json').write_text(json.dumps(out,indent=2)+'\n')
for a,v in out['arms'].items():print(a,v['summary'])
