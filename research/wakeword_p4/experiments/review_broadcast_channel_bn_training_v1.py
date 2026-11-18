"""Terminal BN/artifact audit and transitive comparison to fully reconstructed original ledger."""
import gzip,hashlib,json,math,sys
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/broadcast-channel-bn-training-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 plan_path=R/'4090-broadcast-channel-bn-training-plan-v1.json';plan=read(plan_path)
 for p,h in plan['files'].items():assert sha(p)==h,p
 receipt_path=R/'4090-broadcast-channel-bn-training-execution-v1.json';receipt=read(receipt_path);assert receipt['actual_exit_code']==0
 report_path=R/'4090-broadcast-channel-bn-training-original-v1.json';report=read(report_path)
 assert report['steps']==20000 and report['launch']['pid']==receipt['pid'] and report['plan_sha256']==sha(plan_path)
 for name,h in report['files'].items():assert sha(B/name)==h,name
 # The prior exhaustive reviewer actually reconstructed each full original array.
 reference_review_path=R/'4090-broadcast-residual-training-review-v1.json';reference_review=read(reference_review_path)
 reference_report_path=R/'4090-broadcast-residual-training-original-v1.json';reference_report=read(reference_report_path)
 reference_receipt_path=R/'4090-broadcast-residual-training-execution-v1.json'
 assert reference_review['all_sampling_checks_passed'] and reference_review['steps_verified']==20000
 assert read(R/'4090-broadcast-residual-training-review-execution-v1.json')['actual_exit_code']==0
 assert sha(reference_report_path)==reference_review['report_sha256'] and sha(reference_receipt_path)==reference_review['receipt_sha256']
 assert sha(Path(__file__).with_name('review_broadcast_residual_training_v1.py'))==reference_review['reviewer_sha256']
 reference_log=Path('.local/wakeword/broadcast-residual-training-v1/batches.jsonl.gz')
 assert sha(reference_log)==reference_report['files']['batches.jsonl.gz']
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_broadcast_channel_bn_v1 import CausalBroadcastChannelBN
 torch.manual_seed(726);fresh=CausalBroadcastChannelBN();keys=dict(fresh.named_parameters())
 assert sum(v.numel() for v in keys.values())==8161
 for name,step in [('initial',0),('final',20000)]:
  cp=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=False);assert cp['step']==step and cp['plan_sha256']==sha(plan_path)
  fresh.load_state_dict(cp['model'],strict=True);assert all(torch.isfinite(v).all() for v in cp['model'].values())
  bns=[m for m in fresh.modules() if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))]
  assert len(bns)==11 and all(int(m.num_batches_tracked)==step and (m.running_var>=0).all() for m in bns)
  if step==0:
   assert not cp['optimizer']['state']
   torch.manual_seed(726);initial=CausalBroadcastChannelBN().state_dict();assert all(torch.equal(v,cp['model'][k]) for k,v in initial.items())
  else:
   states=cp['optimizer']['state'];assert len(states)==len(keys)
   assert all(int(v['step'])==20000 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in states.values())
 steps=0
 with gzip.open(reference_log,'rt') as old,gzip.open(B/'batches.jsonl.gz','rt') as current:
  while True:
   a,b=old.readline(),current.readline()
   if not a and not b:break
   assert a and b;steps+=1;a,b=json.loads(a),json.loads(b)
   assert a['step']==b['step']==steps and math.isfinite(b['loss'])
   for key in ['descriptors','features_sha256','positive_sha256','negative_sha256']:assert a[key]==b[key],(steps,key)
   assert len(b['descriptors'])==128
 assert steps==20000
 out=dict(all_sampling_checks_passed=True,steps_verified=steps,descriptor_rows_verified=steps*128,initial_model_reconstructed=True,bn_layers=11,final_moment_updates_per_layer=20000,optimizer_steps=20000,receipt_sha256=sha(receipt_path),report_sha256=sha(report_path),plan_sha256=sha(plan_path),reviewer_sha256=sha(__file__),reference_review_sha256=sha(reference_review_path),reference_log_sha256=sha(reference_log),limits='Complete descriptor/array-hash equality to authenticated ledger previously fully reconstructed from original sampler. Does not reconstruct all arrays again or replay optimizer arithmetic. No recognition or P4 qualification.')
 with (R/'4090-broadcast-channel-bn-training-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
