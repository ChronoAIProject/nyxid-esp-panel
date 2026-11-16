"""Ideal synthetic allocation-mixture compatibility, not model recognition."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_overlap_targets_v1 import targets
from compare_bounded_unit_policy_v1 import reference,raw
from bounded_unit_event_policy_v1 import BoundedUnitEventPolicy
from ordered_generator_matcher_v1 import PATTERN
R=Path('research/wakeword_p4/results');INDEX=R/'4090-causal-unit-training-index-v1.json.gz'
def main():
 def run(seq):
  policy=BoundedUnitEventPolicy();return sum(policy.push(t,c) for t,c in seq)
 base=[(t,255) for t in [*PATTERN,3]]
 assert run(base)==1
 for gap in [1,2]:assert run(base[:5]+[(55,100)]*gap+base[5:])==1
 assert run(base[:5]+[(55,100)]*3+base[5:])==0
 assert run(base[:5]+[(55,255)]+base[5:])==0
 assert run(base[:5]+[(None,255)]+base[5:])==0
 assert run(base[:5]+base[6:])==0
 assert run(base[:-1])==0
 data=json.load(gzip.open(INDEX));rows=[];groups={}
 for i,r in enumerate(data['records']):
  kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if r['rir_tail'] else {};p,ends,mass=targets(r['samples'],r['lead'],r['intervals'],**kwargs)
  # Shift the declared labels only; no features, waveform or hypothetical integer codes.
  for delay in [0,4]:
   q=np.zeros_like(p);q[:delay,0]=1
   if delay:q[delay:]=p[:-delay]
   else:q=p
   labels=q.argmax(1);tokens=[int(v)-1 if v else None for v in labels];confidence=q.max(1);events=dict(raw=raw(tokens),span_only=reference(tokens,np.ones(len(tokens),bool)),full=reference(tokens,confidence>=.5));key=f"delay{delay}/{r['carrier']}/class{r['class_id']}/variant{r['variant']}";g=groups.setdefault(key,dict(recordings=0,raw_hits=0,span_only_hits=0,full_hits=0));g['recordings']+=1
   for mode,v in events.items():g[mode+'_hits']+=bool(v)
   rows.append(dict(index=i,delay_frames=delay,class_id=r['class_id'],carrier=r['carrier'],variant=r['variant'],below_half_frames=int((confidence<.5).sum()),event_samples={k:[int(ends[j]) for j in v] for k,v in events.items()}))
 summary={}
 for delay in [0,4]:
  selected=[x for x in rows if x['delay_frames']==delay];summary[str(delay)]={mode:dict(target_hits=sum(bool(x['event_samples'][mode]) for x in selected if x['class_id']==1),target_count=720,negative_hits=sum(bool(x['event_samples'][mode]) for x in selected if x['class_id']!=1),negative_count=3600) for mode in ['raw','span_only','full']}
 files=[Path(__file__),INDEX,Path('research/wakeword_p4/training/generator_overlap_targets_v1.py'),Path('research/wakeword_p4/experiments/compare_bounded_unit_policy_v1.py')]
 out=dict(schema='bounded-overlap-event-compatibility-v1',files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},summary=summary,groups=groups,records=rows,model_calls=0,optimizer_updates=0,limitations='Ideal declared rectangular/RIR generator allocation mixture, not acoustic posterior or labels. Float>=0.5 reference policy only, no fabricated UINT8 predictions. Existing training metadata only. Analytic bounded-wait cases passed; fixed2frame hypothesis, no threshold sweep or qualification. Raw/span modes explanatory only; negative count is class2–6training renders, not continuous background.')
 with (R/'4090-bounded-overlap-event-compatibility-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
