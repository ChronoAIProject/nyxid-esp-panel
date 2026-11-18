"""Authenticate terminal SAM artifacts and independently reconstruct all main batches."""
import gzip,json,math
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-sam-training-v1/sam');LOCK=R/'4090-binary40-sam-training-plan-v1.json'
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt_path=R/'4090-binary40-sam-training-receipt-v1.json';receipt=json.loads(receipt_path.read_text());assert receipt['exit_code']==0
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 report_path=R/'4090-binary40-sam-sam-training-original-v1.json';report=json.loads(report_path.read_text())
 assert report['steps']==20000 and report['arm']=='sam' and report['plan_sha256']==sha(LOCK) and report['launch']['pid']==receipt['pid']
 for p,h in report['files'].items():assert sha(B/p)==h,p
 checkpoints={}
 for name,step in [('initial',0),('final',20000)]:
  cp=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=False)
  assert cp['step']==step and cp['arm']=='sam' and cp['plan_sha256']==sha(LOCK)
  assert all(torch.isfinite(v).all() for v in cp['model'].values())
  assert cp['geometry']['parameters']==30593
  if step==0:assert not cp['optimizer']['state']
  else:
   states=cp['optimizer']['state'];assert len(states)==len(cp['model'])
   assert all(int(v['step'])==20000 for v in states.values())
   assert all(torch.isfinite(v[k]).all() for v in states.values() for k in ['exp_avg','exp_avg_sq'])
  checkpoints[name]=cp
 parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
 assert parent.keys()==checkpoints['initial']['model'].keys()
 assert all(torch.equal(v,parent[k]) for k,v in checkpoints['initial']['model'].items())
 from causal_source_balanced_batch_v1 import Sampler,digest
 sampler=Sampler(726);steps=0
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as baseline,gzip.open(B/'batches.jsonl.gz','rt') as current:
  while True:
   a=baseline.readline();b=current.readline()
   if not a and not b:break
   assert a and b
   a=json.loads(a);b=json.loads(b);steps+=1
   assert a['step']==b['step']==steps
   arrays,descriptors=sampler.batch()
   assert a['descriptors']==b['descriptors']==descriptors and len(descriptors)==128
   for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],arrays,strict=True):assert a[key]==b[key]==digest(value)
   assert b['batch_audit']==dict(original128_exact=True,appended_rows=0)
   assert all(math.isfinite(b[k]) for k in ['loss','perturbed_loss','gradient_norm']) and b['gradient_norm']>0
   if steps%2000==0:print(json.dumps(dict(verified_steps=steps)),flush=True)
 assert steps==20000
 out=dict(schema='binary40-sam-training-review-v1',all_sampling_checks_passed=True,steps_verified=steps,descriptor_rows_verified=steps*128,initial_original_exact=True,all_initial_final_tensors_finite=True,optimizer_steps=20000,receipt_sha256=sha(receipt_path),report_sha256=sha(report_path),plan_sha256=sha(LOCK),reviewer_sha256=sha(__file__),recognition_scored=False,limits='Batch reconstruction and terminal artifact audit; two-gradient perturbation verified separately in preflight, not independently replayed for all updates. No recognition/P4 claim.')
 with (R/'4090-binary40-sam-training-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
