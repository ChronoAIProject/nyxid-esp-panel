"""Recompute aligned supervision crossings from complete saved training scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from causal_sequence_supervision_v1 import masks
R=Path('research/wakeword_p4/results')
rows=[json.loads(s) for s in Path('.local/wakeword/causal-onset-material-v1/records.jsonl').read_text().splitlines()]
refs={r['id']:r for r in json.loads((R/'4090-causal-command-prefix-reference-map-v1.json').read_text())['records']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'schema':'frozen-history-teacher-sequence-fit-review-v1','arms':{},'limits':'All original synthetic cached presentations, aligned-mask crossings independently recomputed; events taken from authenticated saved records, not independently decoded here. Training fit only, no physicalPCM/generalization/causal attribution/P4 claim.'}
baseline=None
for arm in ['original','control','teacher']:
 path=R/f'4090-frozen-history-teacher-{arm}-sequence-fit-original-v1.json';d=json.loads(path.read_text());assert d['arm']==arm and d['optimizer_updates']==0 and sha(d['trace']['path'])==d['trace']['sha256'];assert len(d['records'])==len(rows)==51840
 groups={};changes=[]
 with np.load(d['trace']['path']) as z:
  prob=z['probability'];assert prob.shape==(51840,282) and np.isfinite(prob).all()
  for i,(m,r) in enumerate(zip(rows,d['records'],strict=True)):
   s=m['source'];end=m['container_end_samples']
   if s['id'] in refs:end=m['lead_samples']+refs[s['id']]['automatic_prefix_end_sample']+(m['container_end_samples']-m['lead_samples']-s['native_samples'])
   assert r['index']==i and r['order']==m['order'] and r['source_id']==s['id'] and r['frames']==m['frames'] and r['label']==s['label'] and r['lead_samples']==m['lead_samples'] and r['aligned_end_samples']==end
   _,pw,_=masks(m['frames'],s['label'],m['lead_samples'],end);valid=len(pw);assert valid==r['valid_calls'];v=prob[i,:valid].astype(np.float64);totals=np.convolve(v,np.ones(5),'valid');eligible=np.convolve((pw>0).astype(int),np.ones(5,dtype=int),'valid')==5;cross=(np.flatnonzero(eligible&(totals>=5*.91796875))+5).tolist();assert cross==r['positive_crossings']
   key=s['engine']+'/'+s['text'];g=groups.setdefault(key,dict(rows=0,positive_crossing_rows=0,valid_event_rows=0,gains=0,losses=0));g['rows']+=1;g['positive_crossing_rows']+=bool(cross);g['valid_event_rows']+=any(e<=valid for e in r['events'])
   if baseline is not None:
    old=baseline[i];assert all(old[k]==r[k] for k in ['index','order','source_id','label','frames','lead_samples','aligned_end_samples','valid_calls']);a=bool(old['positive_crossings']) if s['label'] else any(e<=valid for e in old['events']);b=bool(cross) if s['label'] else any(e<=valid for e in r['events'])
    if a!=b:g['gains']+=b;g['losses']+=a;changes.append(dict(index=i,source_id=s['id'],text=s['text'],engine=s['engine'],speaker_id=s['speaker_id'],label=s['label'],before=a,after=b))
 for key,g in groups.items():assert all(d['groups'][key][k]==g[k] for k in ['rows','positive_crossing_rows','valid_event_rows'])
 if baseline is None:baseline=d['records']
 out['arms'][arm]=dict(all_passed=True,report_sha256=sha(path),groups=groups,changes=changes)
out['all_passed']=True
(R/'4090-frozen-history-teacher-sequence-fit-review-v1.json').write_text(json.dumps(out,indent=2)+'\n')
for arm,d in out['arms'].items():print(arm,json.dumps(d['groups']))
