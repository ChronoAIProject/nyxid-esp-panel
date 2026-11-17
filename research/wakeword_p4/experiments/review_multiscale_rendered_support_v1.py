"""Independent exact-input/label and saved-probability recomputation."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=R/'4090-multiscale-rendered-support-plan-v1.json';report=R/'4090-multiscale-rendered-support-original-v1.json';p=json.loads(plan.read_text());s=json.loads(report.read_text());assert s['plan_sha256']==sha(plan) and s['optimizer_updates']==0
for f,h in p['files'].items():assert sha(f)==h,f
assert sha(s['inputs']['path'])==s['inputs']['sha256'];refs={r['id']:r for r in json.loads((R/'4090-causal-command-prefix-reference-map-v1.json').read_text())['records']};bins={n:np.memmap(n,dtype='<u2',mode='r').reshape(-1,40) for n in {r['feature_binary'] for r in p['rows']}}
out={'schema':'multiscale-rendered-support-review-v1','report_sha256':sha(report),'arms':{},'input_rows_verified':0,'all_passed':False}
with np.load(s['inputs']['path']) as z:
 x=z['features'];pw=z['positive'];nw=z['negative'];assert x.shape==(1080,846,40) and pw.shape==nw.shape==(1080,282)
 for i,row in enumerate(p['rows']):
  frames=row['frames'];raw=bins[row['feature_binary']][row['offset']:row['offset']+frames];expected=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);assert np.array_equal(x[i,:frames],expected) and not x[i,frames:].any()
  end=row['container_end_samples'];ref=refs.get(row['source']['id'])
  if ref:end=row['lead_samples']+ref['automatic_prefix_end_sample']+end-row['lead_samples']-row['source']['native_samples']
  calls=(frames+2)//3;clock=np.arange(1,calls+1)*480+320;pos=(clock>=end+480)&(clock<=end+4320);neg=clock<=row['lead_samples'];pp=np.zeros(282,np.float32);nn=pp.copy();pp[:calls]=pos.astype(np.float32)/pos.sum();nn[:calls]=neg.astype(np.float32)/neg.sum();assert np.allclose(pp,pw[i],atol=1e-8,rtol=0) and np.allclose(nn,nw[i],atol=1e-8,rtol=0);out['input_rows_verified']+=1
 for arm,a in s['arms'].items():
  assert sha(a['trace']['path'])==a['trace']['sha256'];assert a['checkpoint_sha256']==sha(f'.local/wakeword/binary40-multiscale-training-v1/{arm}/final.pt')
  with np.load(a['trace']['path']) as q:prob=q['probability'].copy()
  assert prob.shape==(1080,282) and np.isfinite(prob).all();assert len(a['rows'])==1080;groups={}
  for i,r in enumerate(a['rows']):
   row=p['rows'][i];assert r['order']==row['order'] and r['source_id']==row['source']['id'];v=prob[i].astype(np.float64);cs=np.r_[0,np.cumsum(v)];means=(cs[5:]-cs[:-5])/5;valid=pw[i,4:]>0;peak=float(means[valid].max());hit=bool((means[valid]>=.91796875).any());assert abs(peak-r['maximum_five_mean'])<1e-12 and hit==r['crossing'];assert abs(float((prob[i]*pw[i]).sum())-r['positive_mean'])<1e-7
   g=groups.setdefault(r['text'],{'rows':0,'crossings':0,'minimum_peak':1.0});g['rows']+=1;g['crossings']+=int(hit);g['minimum_peak']=min(g['minimum_peak'],peak)
  assert sum(g['crossings'] for g in groups.values())==a['crossings'];out['arms'][arm]={'groups':groups,'trace_sha256':a['trace']['sha256']}
out['all_passed']=True;out['reviewer_sha256']=sha(__file__);out['limits']='Exact rendered feature and mechanical label reproduction plus independent saved-score metrics only. No independent inference, new exposure, unseen-source/P4 acceptance or causal mechanism proof.'
(R/'4090-multiscale-rendered-support-review-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['arms']))
