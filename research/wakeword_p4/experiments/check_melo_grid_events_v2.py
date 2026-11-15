"""Decoder boundary/state fixtures only; no model numerical equality audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
from melo_grid_event_counts_v2 import grid_event_counts
from uint8_event_policy_v1 import event_steps

def main():
 sums=np.arange(1276,dtype=np.int64)[:,None];k=np.arange(1,1281,dtype=np.int64)[None,:]
 assert np.array_equal(sums/1280>=k/1280,sums>=k)
 assert np.array_equal(sums/1280<(k/1280)*.5,2*sums<k)
 vectors=[np.concatenate([np.zeros(172,np.uint8),np.full(90,255,np.uint8),np.zeros(80,np.uint8),np.full(95,224,np.uint8),np.zeros(20,np.uint8)]),np.random.default_rng(731).integers(0,256,512,dtype=np.uint8)]
 checked=0
 for raw in vectors:
  actual=grid_event_counts(raw,300000,48000)
  for i in range(1280):
   steps=event_steps(raw,(i+1)/1280);times=[min((((3*s+2)*160+479)//480)*480,381600)-81600 for s in steps];real=[t for t in times if t>0]
   expected=[len(real),int(any(t>=48000 for t in real)),len(times)-len(real)]
   assert actual[i].tolist()==expected,(i,actual[i].tolist(),expected);checked+=1
 result=dict(schema='melo-grid-event-analytic-check-v2',event_profiles=2,threshold_profile_pairs=checked,all_possible_five_uint8_sums_and_grid_boundaries_checked=True,priming_refractory_rearm_and_return_clocks_checked=True,independent_python_event_semantics_match=True,pure_event_semantics_only=True,model_or_audio_loaded=False,real_development_scores_consumed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='Artificial UINT8 decoder fixtures and exhaustive attainable window-sum boundaries only. Not CPU/GPU neural numerical agreement or a promotion prerequisite beyond correct event implementation.')
 with Path('research/wakeword_p4/results/4090-melo-grid-event-analytic-check-v2.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
