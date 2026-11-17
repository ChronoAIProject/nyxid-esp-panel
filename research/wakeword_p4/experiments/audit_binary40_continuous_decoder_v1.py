"""Describe fixed decoder blocking in authenticated existing continuous traces."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
roots={'c161':'.local/wakeword/continuous-chatterbox-v1','binary40':'.local/wakeword/binary40-continuous-replay-v1','control64':'.local/wakeword/binary40-capacity-control64-continuous-replay-v1','wide128':'.local/wakeword/binary40-capacity-wide128-continuous-replay-v1'}
files={__file__:sha(__file__)};rows=[];summaries={};timeline_by_gap={}
for arm,root in roots.items():
 p=Path(root)/'result.json';files[str(p)]=sha(p);d=json.loads(p.read_text());assert d['completed']
 for rec in d['results']:
  gap=rec['gap_samples']; timeline=rec['timeline']
  if gap in timeline_by_gap:assert timeline==timeline_by_gap[gap]
  else:timeline_by_gap[gap]=timeline
  model=rec['models']['c161' if arm=='c161' else 'binary40'];t=Path(model['trace_path']);assert sha(t)==model['trace_sha256'];files[str(t)]=sha(t)
  with np.load(t) as z:
   codes=z['uint8']; clocks=z['relative_block_return']; assert codes.dtype==np.uint8
   assert len(codes)==len(clocks) and np.all(np.diff(clocks)>=0)
   total=0;last=-1000;armed=True;emitted=[];calls=[]
   for j,value in enumerate(codes):
    step=j+1;total+=int(value)
    if j>=5:total-=int(codes[j-5])
    if j<4:continue
    if total<587.5 and step-last>=67:armed=True
    high=total>=1175;blocked_cooldown=high and step-last<67;blocked_rearm=high and not armed
    event=high and armed and step-last>=67
    calls.append(dict(sample=int(clocks[j]),step=step,sum5=total,high=high,cooldown=blocked_cooldown,unarmed=blocked_rearm,event=event))
    if event:emitted.append(step);last=step;armed=False
   assert emitted==event_steps(codes,.91796875)==[e['step'] for e in model['events']]
   counter=Counter()
   for clip in timeline:
    if clip['kind']!='clip' or clip['group']!='nyx':continue
    cc=[x for x in calls if clip['start']<x['sample']<=clip['end']];assert cc
    hit=any(x['event'] for x in cc);high=any(x['high'] for x in cc)
    category='emitted' if hit else 'no_crossing' if not high else 'blocked_crossing'
    counter[category]+=1
    rows.append(dict(model=arm,gap_samples=gap,container=clip,category=category,max_sum5=max(x['sum5'] for x in cc),high_calls=sum(x['high'] for x in cc),refractory_blocked_calls=sum(x['cooldown'] for x in cc),unarmed_high_calls=sum(x['unarmed'] for x in cc)))
   assert sum(counter.values())==12;summaries[f'{arm}/{gap}']=dict(counter)
out=dict(schema='binary40-continuous-decoder-audit-v1',files=files,summary=summaries,records=rows,limits='Existing consumed synthetic continuous containers only. State carried through priming/boundaries; fixed original UINT8 decoder reproduced exactly. Blocked-call counts overlap. No-crossing means none in whole clip container, not phonetic timing truth. No inference, optimization, policy changes, human/P4 or command-retention claim.')
(R/'4090-binary40-continuous-decoder-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries,indent=2))
