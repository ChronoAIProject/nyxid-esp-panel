"""Symbolic frame-policy checks, not audio/recognition."""
import gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from ordered_unit_event_policy_v1 import UnitEventPolicy
from ordered_generator_matcher_v1 import PATTERN
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def feed(seq,policy=None):
 p=policy or UnitEventPolicy();return [i for i,(t,c) in enumerate(seq) if p.push(t,c)]
def main():
 assert feed([(t,255) for t in [*PATTERN,3]])==[len(PATTERN)]
 assert not feed([(t,127 if i==5 else 255) for i,t in enumerate([*PATTERN,3])])
 assert not feed([(t,255) for t in PATTERN[:5]]+[(None,255)]*300+[(t,255) for t in [*PATTERN[5:],3]])
 assert not feed([(t,255) for t in [*PATTERN,62,3]])
 assert not feed([(t,255) for t in [*PATTERN[:-1],3]])
 assert not feed([(t,255) for t in PATTERN[:5]]+[(PATTERN[4],255)]*210+[(t,255) for t in [*PATTERN[5:],3]])
 seq=[(t,255) for t in [*PATTERN,3]];assert len(feed(seq*2))==1
 assert len(feed(seq+[(None,0)]*200+seq))==2
 source=R/'4090-joined-generator-token-intervals-v1.json.gz'
 with gzip.open(source,'rt') as f:rows=json.load(f)['records']
 successes=0;durations=[]
 for r in rows:
  # Symbolic ideal frameendlabels directlyfromgeneratorallocation, noacousticposterior.
  iv=r['intervals'];sequence=[];idx=0
  for end in range(400,r['samples']+1,160):
   point=end-1
   while idx+1<len(iv) and point>=iv[idx]['end_sample']:idx+=1
   sequence.append((iv[idx]['token_id'],255))
  hit=bool(feed(sequence));target=r['text'].startswith('Hey Nicks,');assert hit==target;successes+=hit
  if target:durations.append(iv[11]['end_sample']-iv[1]['start_sample'])
 out=dict(schema='ordered-unit-event-policy-symbolic-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/ordered_unit_event_policy_v1.py'),Path('research/wakeword_p4/training/ordered_generator_matcher_v1.py'),source]},confidence_code=128,max_span_frames=200,refractory_frames=200,frame_hop_samples=160,symbolic_sequences=len(rows),target_matches=successes,target_generated_span_samples_minmax=[min(durations),max(durations)],analytic_policy_cases=8,all_passed=True,waveform_recognition_calls=0,policy='Proposedeventprotocol atfixeddiagnosticcode128/256, notselectedoperatingpoint orreplacementv30threshold. ExactargmaxorderedIDs, adjacentrepeatcollapse, boundaryrequired, lowconfidence/unallocatedclears,2smaxspan/2scooldown. Greedymatcherdoesnotintegrateposteriors/beamsearch andmaymissshorttokenpredictions. Originalpaddingtoken0 distinctfromunallocatedNone. Symbolicidealframeendlabelsnottrainedoutputs orsofttargetCE consistency proof; noqualification. Freshdevelopmentselection mustlaterpreregisterallowedoperatingpoints beforeuse, nottuneonconsumeddata. Barepositivesnotrepresentedbycommandonlytrainingyet.')
 with (R/'4090-ordered-unit-event-policy-symbolic-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,symbolic_sequences=len(rows),target_matches=successes)))
if __name__=='__main__':main()
