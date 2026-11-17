"""Independent saved-array event and complete inventory audit, no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/multiscale-float-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def decode(values,integer):
 # Independent window reduction instead of producer's convolution.
 sums=np.lib.stride_tricks.sliding_window_view(values.astype(np.float64),5).sum(axis=1)
 threshold=1175 if integer else .91796875*5
 last=-1000;armed=True;events=[]
 for i,total in enumerate(sums,5):
  if total<threshold*.5 and i-last>=67:armed=True
  if total>=threshold and armed and i-last>=67:events.append(i);last=i;armed=False
 return events
p=read(R/'4090-multiscale-float-original-v1.json');assert p['plan_sha256']==sha(R/'4090-multiscale-float-plan-v1.json')
expected=dict(builtin=36,daily=96,continuous=2,primed=372,jenny=408,joe=360,melo=474,mac=12,context=3)
reports={}
for arm in ['control','multiscale']:
 ref=p['arms'][arm];assert sha(ref['path'])==ref['sha256'];report=read(ref['path']);assert report['optimizer_updates']==0 and report['native_parity_max_error']<1e-4
 assert report['loaded_checkpoint_sha256']==sha(f'.local/wakeword/binary40-multiscale-training-v1/{arm}/final.pt')
 rows=report['rows'];assert len(rows)==1763
 saved=[json.loads(s) for s in (B/arm/'records.jsonl').read_text().splitlines()];assert saved==rows
 paths=set();counts={};calls=0
 for row in rows:
  trace=row['trace'];assert trace['path'] not in paths;paths.add(trace['path']);assert sha(trace['path'])==trace['sha256'];assert Path(trace['path']).stat().st_size==trace['bytes']
  with np.load(trace['path']) as z:
   f=z['probability'];codes=z['uint8'];clock=z['relative_block_return'];assert len(f)==len(codes)==len(clock);assert codes.dtype==np.uint8 and np.isfinite(f).all() and ((f>=0)&(f<=1)).all();assert (np.diff(clock)>=0).all()
   for name,v,integer in [('float',f,False),('int8',codes,True)]:
    events=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in decode(v,integer)];assert events==row['events'][name],(arm,row['cohort'],row['index'],name)
   calls+=len(f)
  counts[row['cohort']]=counts.get(row['cohort'],0)+1
 assert counts==expected,(counts,expected)
 assert {str(x) for x in (B/arm).glob('*.npz')}==paths
 reports[arm]=dict(streams=len(rows),calls=calls,cohorts=counts,trace_members_verified=len(paths),report_sha256=ref['sha256'])
out=dict(schema='multiscale-float-independent-review-v1',all_passed=True,arms=reports,original_report_sha256=sha(R/'4090-multiscale-float-original-v1.json'),reviewer_sha256=sha(__file__),limits='Full local saved-array/inventory review only. Archive/member and saved-INT8 identity binding require separate checks; no P4 or acceptance claim.')
with (R/'4090-multiscale-float-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
