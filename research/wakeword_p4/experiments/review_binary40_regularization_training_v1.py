"""Terminal/checkpoint and exhaustive original/main/paired provenance review."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/binary40-regularization-training-v1');PLAN=R/'4090-binary40-regularization-training-plan-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt=R/'4090-binary40-regularization-training-terminal-v1.json';terminal=json.loads(receipt.read_text());assert terminal['status']=='completed-await-independent-review' and terminal['plan_sha256']==sha(PLAN)
 assert [r['arm'] for r in terminal['stages']]==['control','decay'] and all(r['actual_exit_code']==0 for r in terminal['stages'])
 plan=json.loads(PLAN.read_text())
 for f,h in plan['files'].items():assert sha(f)==h,f
 for row in terminal['stages']:assert sha(row['log_path'])==row['log_sha256']
 original=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=False);arms={}
 for arm,weight_decay in plan['arms'].items():
  path=R/f'4090-binary40-regularization-{arm}-training-original-v1.json';report=json.loads(path.read_text());assert report['steps']==report['optimizer_updates']==20000 and report['teacher_unchanged'] and report['teacher_disconnected'];assert report['launch']['plan_sha256']==sha(PLAN)
  for name,h in report['files'].items():assert sha(ROOT/arm/name)==h
  initial=torch.load(ROOT/arm/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(ROOT/arm/'final.pt',map_location='cpu',weights_only=False)
  assert initial['step']==0 and final['step']==20000 and final['coefficient']==1 and final['weight_decay']==weight_decay and final['plan_sha256']==sha(PLAN)
  assert set(initial['model'])==set(final['model'])==set(original['model'])
  assert all(torch.equal(initial['model'][k],v) for k,v in original['model'].items())
  assert all(torch.isfinite(v).all() for v in final['model'].values())
  assert report['launch']['arm']==arm and report['launch']['coefficient']==1 and report['launch']['weight_decay']==weight_decay
  assert initial['arm']==final['arm']==arm
  assert initial['initial_checkpoint_sha256']==final['initial_checkpoint_sha256']==sha(plan['initial_checkpoint'])
  # Reconstruct rank-based grouping independently from the final named tensors.
  from causal_binary40_v1 import Binary40
  native=Binary40();assert list(dict(native.named_parameters()))==list(final['model'])
  names=list(final['model']);matrix=[k for k in names if final['model'][k].ndim>=2];vectors=[k for k in names if final['model'][k].ndim<2]
  groups=final['optimizer']['param_groups'];assert len(groups)==2
  assert groups[0]['params']==list(range(len(matrix))) and groups[1]['params']==list(range(len(matrix),len(names)))
  assert groups[0]['weight_decay']==weight_decay and groups[1]['weight_decay']==0
  for group in groups:
   assert group['lr']==.001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['foreach'] is False
   assert not group['amsgrad'] and not group['maximize']
  assert set(final['optimizer']['state'])==set(range(len(names)))
  assert all(float(s['step'])==20000 for s in final['optimizer']['state'].values())
  assert all(torch.isfinite(v).all() for state in final['optimizer']['state'].values() for v in state.values() if torch.is_tensor(v))
  arms[arm]=dict(training_report_sha256=sha(path),final_checkpoint_sha256=sha(ROOT/arm/'final.pt'),initial_checkpoint_sha256=sha(ROOT/arm/'initial.pt'),teacher_unchanged_reported=True,optimizer_updates=20000,teacher_kl_coefficient=1,matrix_weight_decay=weight_decay,matrix_parameter_names=matrix,excluded_vector_parameter_names=vectors,optimizer_groups_verified=True)
 with gzip.open(R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz','rt') as f:schedule=[json.loads(x) for x in f]
 count=0;seen=set()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as orig,gzip.open(ROOT/'control/batches.jsonl.gz','rt') as control,gzip.open(ROOT/'decay/batches.jsonl.gz','rt') as decay:
  for step in range(1,20001):
   old=json.loads(next(orig));a=json.loads(next(control));b=json.loads(next(decay));assert old['step']==a['step']==b['step']==step
   for key in ['features_sha256','positive_sha256','negative_sha256']:assert a[key]==b[key]==old[key]
   descriptor=hashlib.sha256(json.dumps(old['descriptors'],sort_keys=True).encode()).hexdigest();assert a['main_descriptor_sha256']==b['main_descriptor_sha256']==descriptor
   assert a['pair_indices']==b['pair_indices']==schedule[step-1]['inventory_indices'];assert a['pair_tensor_hashes']==b['pair_tensor_hashes'] and len(a['pair_tensor_hashes'])==4;seen.update(a['pair_indices'])
   for row in [a,b]:
    assert all(np.isfinite(row[k]) for k in ['main_bce','paired_bce','kl','total'])
    assert np.isclose(row['total'],row['main_bce']+row['paired_bce']+row['kl'],rtol=2e-6,atol=1e-7)
   count+=1
  assert orig.readline()==control.readline()==decay.readline()==''
 assert count==20000 and len(seen)==4320
 out=dict(schema='binary40-regularization-training-review-v1',all_passed=True,every_batch_identical_to_original=True,every_paired_batch_identical_between_arms=True,pair_indices_match_fixed_schedule=True,steps=20000,unique_paired_sources=len(seen),terminal_receipt_sha256=sha(receipt),plan_sha256=sha(PLAN),arms=arms,limits='Exhaustive saved main hashes/descriptors checked against original and paired tensor hashes between arms; no new full tensor generation. Teacher immutability enforced in original training code and reported, not reconstructed intermediate tensors. Recognition/INT8/P4 unproven.')
 (R/'4090-binary40-regularization-training-review-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
