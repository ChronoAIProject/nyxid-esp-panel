"""Authenticate complete saved float traces and independently decode events."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def decode(v):
 history=[];last=-1000;armed=True;events=[]
 for step,value in enumerate(v,1):
  history.append(float(value))
  if len(history)>5:history.pop(0)
  if len(history)<5:continue
  mean=sum(history)/5
  if mean<.91796875/2 and step-last>=67:armed=True
  if armed and mean>=.91796875 and step-last>=67:events.append(step);last=step;armed=False
 return events
out={'schema':'frozen-history-teacher-float-review-v1','arms':{},'limits':'Saved complete consumed traces only; no new inference, tuning, acceptance or P4 evidence.'}
for arm in ['control','teacher']:
 report=R/f'4090-frozen-history-teacher-{arm}-float-replay-original-v1.json';d=read(report)
 assert d['teacher_arm']==arm and d['optimizer_updates']==0
 plan=R/f'4090-frozen-history-teacher-{arm}-float-replay-plan-v1.json';assert sha(plan)==d['plan_sha256']
 assert set(d['cohorts'])==set(read(plan)['cohorts'])
 paths=set();total=0;cohorts={}
 for c,spec in d['cohorts'].items():
  assert sha(spec['path'])==spec['sha256'];v=read(spec['path']);assert v['completed'] and v['plan_sha256']==d['plan_sha256'];assert v['groups']==spec['groups']
  mismatches=[]
  for row in v['records']:
   p=Path(row['trace']['path']);assert str(p) not in paths;paths.add(str(p));assert sha(p)==row['trace']['sha256'] and p.stat().st_size==row['trace']['bytes']
   old=row['saved_int8_trace'];assert sha(old['path'])==old['sha256']
   with np.load(p) as z:
    values=z['float_probability'];clock=z['relative_block_return'];assert np.isfinite(values).all() and len(values)==len(clock)
    events=[dict(step=s,relative_sample=int(clock[s-1])) for s in decode(values)]
   assert events==row['events']['float']
   if row['hits']['float']!=row['hits']['int8']:mismatches.append(dict(index=row['index'],phase=row['phase'],source=row['source'],hits=row['hits']))
   total+=1
  cohorts[c]=dict(rows=len(v['records']),groups=v['groups'],float_int8_hit_differences=mismatches)
 assert total==1760
 out['arms'][arm]=dict(all_passed=True,rows=total,unique_traces=len(paths),report_sha256=sha(report),cohorts=cohorts)
out['all_passed']=True
(R/'4090-frozen-history-teacher-float-review-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({a:dict(rows=v['rows'],unique_traces=v['unique_traces']) for a,v in out['arms'].items()}))
