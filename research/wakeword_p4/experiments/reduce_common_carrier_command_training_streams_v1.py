"""Audit complete training traces and retain all-grid score support, no threshold selection."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from probe_common_carrier_command_training_streams_v1 import LOCK,OUT,sha
R=Path('research/wakeword_p4/results')
def main():
 terminal=json.loads((OUT/'terminal.json').read_text());p=json.loads(LOCK.read_text())
 assert terminal['execution_sha256']==sha(LOCK) and terminal['records']==9720
 grid=json.loads((R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json').read_text())['recognition']['grid'];assert len(grid)==138
 seen=set();rows=[];groups={}
 for line in (OUT/'records.jsonl').read_text().splitlines():
  rec=json.loads(line);i=rec['index'];condition=rec['condition'];assert (i,condition) not in seen;seen.add((i,condition))
  assert rec['row']==p['records'][i]['row'] and rec['original_index']==p['records'][i]['original_index']
  assert sha(rec['trace'])==rec['trace_sha256'] and rec['repeat_exact'];r=rec['row']
  with np.load(rec['trace'],allow_pickle=False) as a:
   v=a['probability'];assert v.dtype==np.float32 and v.tobytes()==a['probability_repeat'].tobytes()
   n=(max(0,(r['samples']+159)//160-2)+2)//3;assert len(v)==n==rec['steps'] and np.isfinite(v).all() and np.all((v>=0)&(v<=1))
   support=(3*np.arange(1,n+1,dtype=np.int64)+2)*160;assert np.array_equal(support,a['support_unclamped']);clocks=np.minimum(support,r['samples']);assert np.array_equal(clocks,a['sample_end'])
   means=np.array([math.fsum(float(z) for z in v[j-4:j+1])/5 for j in range(4,n)])
   boundary=r['generated_prefix_end_sample']
   after=means[clocks[4:]>=boundary] if boundary is not None else np.array([])
   row=dict(index=i,condition=condition,text=r['text'],conditioning_id=r['conditioning_id'],label=r['label'],variant=r['variant'],maximum_five_mean=float(max(means)),maximum_after_generated_boundary=float(max(after)) if len(after) else None,trace_sha256=rec['trace_sha256'])
  rows.append(row);groups.setdefault((condition,r['anchor_text'],r['carrier']),[]).append(row)
 assert len(rows)==9720 and seen=={(i,c) for i in range(4860) for c in ('control','command_context')}
 summaries=[]
 for (condition,text,carrier),items in sorted(groups.items()):
  values=[r['maximum_five_mean'] for r in items];assert len(items)==180
  summaries.append(dict(condition=condition,text=text,carrier=carrier,clips=len(items),label=items[0]['label'],maximum_five_quantiles={str(q):float(np.quantile(values,q)) for q in (0,.05,.25,.5,.75,.95,1)},all_grid_score_support=[dict(threshold=t,clips_with_any_five_mean_at_or_above=sum(v>=t for v in values)) for t in grid]))
 result=dict(execution_sha256=sha(LOCK),terminal_sha256=sha(OUT/'terminal.json'),records_sha256=sha(OUT/'records.jsonl'),audited_traces=9720,all_repeat_bytes_exact=True,groups=summaries,rows=rows,threshold_selected=False,limits='Training-source descriptive score support only; max-five crossing is NOT latched/refractory event count or complete-phrase recognition. No background exposure/FAR measurement here; generated boundary not phonetic truth. Entire original grid retained without point selection; no freshdev/acceptance.')
 with (R/'4090-common-carrier-command-training-streams-result-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(audited_traces=len(rows),groups=[{k:v for k,v in g.items() if k!='all_grid_score_support'} for g in summaries]),indent=2))
if __name__=='__main__':main()
