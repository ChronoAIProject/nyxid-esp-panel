"""Authenticate final broadcast-residual artifacts and reconstruct every original batch."""
import gzip,json,math,sys
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/broadcast-residual-training-v1');LOCK=R/'4090-broadcast-residual-training-plan-v1.json'
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt_path=R/'4090-broadcast-residual-training-execution-v1.json';receipt=json.loads(receipt_path.read_text());assert receipt['actual_exit_code']==0
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 report_path=R/'4090-broadcast-residual-training-original-v1.json';report=json.loads(report_path.read_text());assert report['steps']==20000 and report['plan_sha256']==sha(LOCK) and report['launch']['pid']==receipt['pid']
 for p,h in report['files'].items():assert sha(B/p)==h,p
 cps={}
 for name,step in [('initial',0),('final',20000)]:
  cp=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=False);assert cp['step']==step and cp['plan_sha256']==sha(LOCK)
  assert all(torch.isfinite(v).all() for v in cp['model'].values()) and sum(v.numel() for v in cp['model'].values())==7457
  states=cp['optimizer']['state']
  if step==0:assert not states
  else:
   assert len(states)==len(cp['model']) and all(int(v['step'])==20000 for v in states.values())
   assert all(torch.isfinite(v[k]).all() for v in states.values() for k in ['exp_avg','exp_avg_sq'])
  cps[name]=cp
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_broadcast_residual_v1 import CausalBroadcastResidual
 torch.manual_seed(726);fresh=CausalBroadcastResidual().state_dict();assert fresh.keys()==cps['initial']['model'].keys();assert all(torch.equal(v,cps['initial']['model'][k]) for k,v in fresh.items())
 from causal_source_balanced_batch_v1 import Sampler,digest
 sampler=Sampler(726);steps=0
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as baseline,gzip.open(B/'batches.jsonl.gz','rt') as current:
  while True:
   a=baseline.readline();b=current.readline()
   if not a and not b:break
   assert a and b;a=json.loads(a);b=json.loads(b);steps+=1;assert a['step']==b['step']==steps and math.isfinite(b['loss'])
   arrays,descriptors=sampler.batch();assert a['descriptors']==b['descriptors']==descriptors and len(descriptors)==128
   for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],arrays,strict=True):assert a[key]==b[key]==digest(value)
   if steps%2000==0:print(json.dumps(dict(verified_steps=steps)),flush=True)
 assert steps==20000
 out=dict(all_sampling_checks_passed=True,steps_verified=steps,descriptor_rows_verified=steps*128,initial_model_reconstructed=True,all_initial_final_tensors_finite=True,optimizer_steps=20000,receipt_sha256=sha(receipt_path),report_sha256=sha(report_path),plan_sha256=sha(LOCK),reviewer_sha256=sha(__file__),limits='Full original batch reconstruction and terminal artifact audit; optimizer arithmetic not replayed. No recognition/P4 qualification.')
 with (R/'4090-broadcast-residual-training-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
