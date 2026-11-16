"""Metadata scheduling/balance/recovery check, no audio or optimization."""
import collections,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from causal_unit_sampler_v1 import Sampler,INDEX,CARRIERS
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=Sampler();counts=collections.Counter();seen=set();digest=hashlib.sha256()
 for step in range(1,5001):
  d=s.descriptors();assert len(d)==24 and collections.Counter((x['carrier'],x['class_id']) for x in d)==collections.Counter((c,l) for c in CARRIERS for l in range(1,7));digest.update(json.dumps(d,sort_keys=True).encode())
  for x in d:counts[x['carrier']]+=1;seen.add(x['index'])
  if step==2500:
   state=s.state();clone=Sampler(999);clone.restore(state);assert clone.descriptors()==SamplerProbe(s)
 out=dict(schema='causal-unit-sampler-contract-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/causal_unit_sampler_v1.py'),INDEX]},batches=5000,batch_recordings=24,carrier_record_draws=dict(counts),distinct_records=len(seen),all_cells_balanced_per_batch=True,rng_restore_passed=True,descriptors_sha256=digest.hexdigest(),policy='Metadataonly schedulecheck,5000notyettrainingbudget. Complete recordingindices, noPCM/features oroptimization. One matchedsixclasscell percarrier eachbatch, independentcarrierRNG, sharedpermutation; noequalphysicalspeakerclaim. Randomcellwithreplacement, notepochcoverageguarantee. Unknownrealbackgroundnotincluded/mislabeled. Padding/masks/lossbalance/trainingrecipe remainseparate.')
 with (R/'4090-causal-unit-sampler-contract-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
def SamplerProbe(s):
 state=s.state();d=s.descriptors();s.restore(state);return d
if __name__=='__main__':main()
