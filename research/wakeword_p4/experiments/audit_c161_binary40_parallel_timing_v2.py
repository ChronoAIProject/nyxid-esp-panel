"""Saved-event timing and suppression accounting; no inference or parameter search."""
import hashlib,json
from collections import Counter
from pathlib import Path
R=Path('research/wakeword_p4/results')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=load(R/'4090-c161-binary40-parallel-timing-plan-v2.json')
 for p,h in plan['files'].items():assert sha(p)==h,p
 review=load(R/'4090-c161-binary40-parallel-runtime-review-v1.json');assert review['all_passed'] and review['unique_traces']==1760
 refpath=R/'4090-c161-binary40-parallel-events-original-v2.json'
 runtimeplan=load(R/'4090-c161-binary40-parallel-runtime-plan-v1.json');assert runtimeplan['files'][str(refpath)]==sha(refpath)
 ref=load(refpath);details=[];suppressed=[];summaries={};rows=0
 for cohort,group in ref['cohorts'].items():
  comparisons=[];counts=Counter()
  for row in group['records']:
   rows+=1
   # Independent sorted event arbitration, retain exact identity and suppressor.
   candidates=sorted([dict(e,model=m) for m,events in row['saved'].items() for e in events],key=lambda e:(e['step'],0 if e['model']=='c161' else 1))
   kept=[];dropped=[];last=None
   for e in candidates:
    if last is None or e['step']-last['step']>=67:kept.append(e);last=e
    else:
     dropped.append(e);suppressed.append(dict(cohort=cohort,index=row['index'],phase=row['phase'],event=e,suppressor=last.copy(),distance_calls=e['step']-last['step']))
   assert kept==row['emitted'] and dropped==row['suppressed']
   counts.update('suppressed_'+e['model'] for e in dropped)
   if cohort=='continuous':
    windows=[dict(key=t['index'],start=t['start'],end=t['end'],reference='generated Nyx clip container; not phonetic/command onset') for t in row['timeline'] if t['kind']=='clip' and t['group']=='nyx']
   else:
    s=row['source'];target=s.get('target_id')=='hey_nyx' or s.get('target')=='hey_nyx' or s.get('phrase_id')=='nyx' or s.get('group')=='nyx'
    windows=[dict(key='whole_real_stream',start=0,end=None,reference='nonnegative real-stream event, original clip-hit convention; not phonetic/command onset')] if row['phase']=='synthetic' and target else []
   for w in windows:
    def inside(e):return (e['relative_sample']>=0 if w['end'] is None else w['start']<e['relative_sample']<=w['end'])
    a=[e for e in row['saved']['c161'] if inside(e)];b=[e for e in kept if inside(e)]
    counts['target_windows']+=1;counts['c161_hit']+=bool(a);counts['parallel_hit']+=bool(b)
    if a and not b:counts['lost_c161_window']+=1
    if a and b:
     ae=min(a,key=lambda e:e['step']);be=min(b,key=lambda e:e['step']);delta=be['relative_sample']-ae['relative_sample'];comparisons.append(delta)
     details.append(dict(cohort=cohort,index=row['index'],window=w,c161_first=ae,parallel_first=be,parallel_minus_c161_samples=delta,parallel_minus_c161_ms=delta/16))
   # Retain every suppressed event including prime/negative/background; no relabeling.
  counts['paired_windows']=len(comparisons)
  summaries[cohort]=dict(counts=counts,paired_first_return_delta_samples=dict(min=min(comparisons),max=max(comparisons),earlier=sum(x<0 for x in comparisons),equal=sum(x==0 for x in comparisons),later=sum(x>0 for x in comparisons)) if comparisons else None)
 assert rows==1760
 out=dict(schema='c161-binary40-parallel-timing-original-v2',plan_sha256=sha(R/'4090-c161-binary40-parallel-timing-plan-v2.json'),all_saved_arbitration_exact=True,rows=rows,cohorts=summaries,paired_target_windows=details,suppressed_events=suppressed,limits='Consumed sources only. No inference, training, policy change or independent exposure. Nonnegative clip and continuous container timing is not phonetic delay, command onset, buffer retention or P4 latency. Earlier B40 events may suppress later C161 in future data; finite coverage does not guarantee monotonic retention.')
 with (R/'4090-c161-binary40-parallel-timing-original-v2.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(rows=rows,cohorts=summaries,suppressed_events=len(suppressed)),indent=2))
if __name__=='__main__':main()
