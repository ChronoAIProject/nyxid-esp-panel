"""Validate paired final artifacts and every saved batch descriptor before replay."""
import hashlib,json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path("research/wakeword_p4/training").resolve()))
from causal_conditioning_risk_weights_v1 import GroupRisk
from causal_conditioning_balanced_sampler_v1 import Sampler

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 roots={a:Path('.local/wakeword/causal-conditioning-'+a+'-v1') for a in ['uniform','allocation']}
 for root in roots.values():assert (root/'result.json').exists(),f'Not completed: {root}'
 reports={a:json.loads((r/'result.json').read_text()) for a,r in roots.items()}
 for a,r in roots.items():
  report=reports[a];assert report['steps']==5000 and report['provenance']['arm']==a
  for name,h in report['artifacts'].items():assert sha(r/name)==h
 initial={a:torch.load(r/'initial.pt',map_location='cpu',weights_only=False) for a,r in roots.items()}
 assert initial['uniform']['model'].keys()==initial['allocation']['model'].keys()
 assert all(torch.equal(v,initial['allocation']['model'][k]) for k,v in initial['uniform']['model'].items())
 count=0;descriptor_hash=hashlib.sha256();risk=GroupRisk();sampler=Sampler(726)
 with (roots['uniform']/'training.jsonl').open() as f,(roots['allocation']/'training.jsonl').open() as g:
  for count,(left,right) in enumerate(zip(f,g,strict=True),1):
   a=json.loads(left);b=json.loads(right);assert a['step']==b['step']==count
   assert a['descriptors']==b['descriptors'] and a['group']==b['group'];assert a['multiplier']==1.
   expected=sampler.descriptors()
   for actual,reference in zip(a['descriptors'],expected,strict=True):
    assert {k:actual[k] for k in reference}==reference
   assert b['multiplier']==1.
   assert abs(b['loss']-b['unweighted_loss']*b['multiplier'])<=1e-5*max(1.,abs(b['loss']))

   assert all(torch.isfinite(torch.tensor(row[k])) for row in [a,b] for k in ['loss','unweighted_loss','gradient_norm','multiplier'])
   descriptor_hash.update((json.dumps(a['descriptors'],sort_keys=True)+'\n').encode())
 assert count==5000
 final=torch.load(roots['allocation']/'final.pt',map_location='cpu',weights_only=False);assert final['step']==5000 and final['group_risk']==GroupRisk().state()
 out=dict(uniform_multiplier_verified=True,all_sampling_descriptors_reconstructed=True,schema='allocation-margin-training-audit-v1',steps=count,all_passed=True,identical_initial_model=True,identical_all_batch_descriptors=True,descriptor_sha256=descriptor_hash.hexdigest(),inputs={str(r/'result.json'):sha(r/'result.json') for r in roots.values()},script_sha256=sha(__file__),limits='Training provenance audit only, not recognition/INT8/fresh acceptance.')
 p=Path('research/wakeword_p4/results/4090-allocation-margin-training-audit-v1.json')
 with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
