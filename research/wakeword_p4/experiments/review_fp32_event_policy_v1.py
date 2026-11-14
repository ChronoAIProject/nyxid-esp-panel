"""Constructed event-boundary/invalid-input and exact code-lattice compatibility checks."""
import hashlib,json
from pathlib import Path
import numpy as np
from fp32_event_policy_v1 import event_steps
from uint8_event_policy_v1 import event_steps as integer_events
from hey_development_metrics_v1 import GRID
def main():
 checks=[]
 for values,threshold,want in [([1.]*4,1.,[]),([1.]*5,1.,[5]),([1.]*200,.9,[5]),([1.]*5+[0.]*67+[1.]*5,.9,[5,77]),([.5]*5,.5,[5]),([float(np.nextafter(.5,0))]*5,.5,[])]:
  assert event_steps(values,threshold)==want;checks.append('event-boundary')
 rejected=0
 for values,threshold in [([np.nan],.5),([np.inf],.5),([-0.1],.5),([1.1],.5),([255],.5),([True],.5),([.5],0),([.5],np.nan),([.5],True)]:
  try:event_steps(values,threshold)
  except ValueError:rejected+=1
  else:raise AssertionError('Invalid input accepted')
 rng=np.random.default_rng(20261114);parity=0
 for _ in range(10):
  codes=rng.integers(0,256,512,dtype=np.uint8)
  for threshold in GRID:
   assert event_steps(codes.astype(np.float64)/256,threshold)==integer_events(codes,threshold);parity+=1
 source=Path(__file__).with_name('fp32_event_policy_v1.py')
 p=Path('research/wakeword_p4/results/4090-fp32-event-policy-constructed-v1.json')
 p.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),boundary_checks=len(checks),invalid_checks=rejected,exact_code_lattice_comparisons=parity,grid=list(GRID),all_passed=True,limits='Constructed traces only. Exact code-lattice parity does not quantize actual float scores or prove recognition. No model/media access.'),indent=2)+'\n');print({'boundary':len(checks),'invalid':rejected,'parity':parity})
if __name__=='__main__':main()
