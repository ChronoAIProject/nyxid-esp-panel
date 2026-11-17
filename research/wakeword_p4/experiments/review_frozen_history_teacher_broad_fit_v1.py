"""Independent saved-score retention review over all original broad bare rows."""
import hashlib,json
from pathlib import Path
import numpy as np
from causal_sequence_supervision_v1 import masks
R=Path('research/wakeword_p4/results')
meta=[r for r in json.loads(Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json').read_text()) if r['source_pair_key']['carrier']=='bare' and r['text'] in ('Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')]
assert len(meta)==31644
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'schema':'frozen-history-teacher-broad-fit-review-v1','arms':{},'limits':'All31644cached training bare rows only, original zero-cache startup. Not physical PCM, new exposure, causal forgetting attribution, or generalization/P4. Exact source transitions preserved.'}
baseline=None
for arm in ['original','control','teacher']:
 p=R/('4090-binary40-broad-fit-original-v1.json' if arm=='original' else f'4090-frozen-history-teacher-{arm}-broad-fit-original-v1.json');d=json.loads(p.read_text());assert sha(d['trace']['path'])==d['trace']['sha256'];groups={};transitions=[]
 with np.load(d['trace']['path']) as z:
  prob=z['probability'];assert prob.shape==(31644,282) and np.isfinite(prob).all()
  for i,(row,m) in enumerate(zip(d['records'],meta,strict=True)):
   for key in ['source_id','offset','variant','engine','conditioning_id','text','label']:assert row[key]==m[key]
   assert row['index']==i
   _,pw,_=masks(m['frames'],m['label'],int((m['lead_frame']+2)*160) if m['label'] else None,int(m['generated_container_end_sample']) if m['label'] else None)
   valid=len(pw);v=prob[i,:valid];s=np.convolve(v.astype(np.float64),np.ones(5),'valid');eligible=np.convolve((pw>0).astype(int),np.ones(5,dtype=int),'valid')==5
   cross=(np.flatnonzero(eligible&(s>=5*.91796875))+5).tolist();assert cross==row['positive_crossings'] and valid==row['valid_calls']
   g=groups.setdefault(row['engine']+'/'+row['text'],dict(rows=0,positive_crossing_rows=0,valid_event_rows=0));g['rows']+=1;g['positive_crossing_rows']+=bool(cross);g['valid_event_rows']+=any(e<=valid for e in row['events'])
   if baseline is not None:
    old=baseline[i];assert all(old[k]==row[k] for k in ['index','source_id','offset','variant','engine','conditioning_id','text','label','valid_calls'])
    before=bool(old['positive_crossings']) if row['label'] else any(e<=valid for e in old['events']);after=bool(cross) if row['label'] else any(e<=valid for e in row['events'])
    if before!=after:transitions.append(dict(index=i,source_id=row['source_id'],variant=row['variant'],engine=row['engine'],text=row['text'],label=row['label'],before=before,after=after))
 for k,g in groups.items():assert all(d['groups'][k][n]==v for n,v in g.items())
 if baseline is None:baseline=d['records']
 out['arms'][arm]=dict(all_passed=True,report_sha256=sha(p),groups=groups,transitions_from_original=transitions)
out['all_passed']=True
(R/'4090-frozen-history-teacher-broad-fit-review-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({a:{'groups':v['groups'],'transitions':len(v['transitions_from_original'])} for a,v in out['arms'].items()}))
