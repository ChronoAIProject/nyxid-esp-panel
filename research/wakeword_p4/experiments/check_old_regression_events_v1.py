"""Verify event transport against original scalar policy, no model/data scoring."""
import json,hashlib
from pathlib import Path
import numpy as np
from uint8_event_policy_v1 import event_steps
from hey_development_metrics_v1 import GRID
from old_regression_events_v1 import events

def scalar_float(x,t):
 result=[];last=-1000;armed=True
 for s in range(5,len(x)+1):
  mean=sum(float(v) for v in x[s-5:s])/5
  if mean<t*.5 and s-last>=67:armed=True
  if armed and mean>=t and s-last>=67:result.append(s);last=s;armed=False
 return result

def main():
 rng=np.random.default_rng(20261114);inputs=[rng.integers(0,256,1200,dtype=np.uint8) for _ in range(4)]
 inputs += [np.zeros(300,np.uint8),np.full(300,255,np.uint8),np.tile(np.r_[np.full(67,255,np.uint8),np.zeros(67,np.uint8)],4),np.r_[np.full(5,255,np.uint8),np.zeros(62,np.uint8),np.full(250,255,np.uint8)]]
 checks=0
 for x in inputs:
  for t in GRID:
   assert events(x,t,'uint8')==event_steps(x,t)
   f=(x.astype(np.float32)/np.float32(256)).astype(np.float32)
   assert events(f,t,'float')==scalar_float(f,t)==event_steps(x,t)
   checks+=2
 for _ in range(4):
  f=rng.uniform(.00033535,.99966465,1200).astype(np.float32)
  for t in GRID:assert events(f,t,'float')==scalar_float(f,t);checks+=1
 p=Path('research/wakeword_p4/results/4090-old-regression-events-check-v1.json')
 with p.open('x') as out:json.dump(dict(checks=checks,grid_points=len(GRID),all_exact=True,source_sha256=hashlib.sha256(Path(__file__).with_name('old_regression_events_v1.py').read_bytes()).hexdigest(),limits='Constructed arrays only; float events use original unrounded float32 probabilities, no UINT8 conversion. No detector or development scoring/selection.'),out,indent=2);out.write('\n')
 print(checks)
if __name__=='__main__':main()
