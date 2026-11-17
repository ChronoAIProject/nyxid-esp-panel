"""Terminal/checkpoint and exhaustive original/main/paired provenance review."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/external-representation-training-v1');PLAN=R/'4090-external-representation-training-plan-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt=R/'4090-external-representation-training-terminal-v1.json';terminal=json.loads(receipt.read_text());assert terminal['status']=='completed-await-independent-review' and terminal['plan_sha256']==sha(PLAN)
 assert [r['arm'] for r in terminal['stages']]==['control','feature'] and all(r['actual_exit_code']==0 for r in terminal['stages'])
 plan=json.loads(PLAN.read_text())
 for f,h in plan['files'].items():assert sha(f)==h,f
 for row in terminal['stages']:assert sha(row['log_path'])==row['log_sha256']
 original=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=False);arms={}
 targets_report=json.loads((R/'4090-external-representation-target-cache-original-v1.json').read_text())
 with np.load(targets_report['arrays']['path'],allow_pickle=False) as cache:
  delta=cache['raw_prefix'].astype(np.float64)-cache['training_mean']
  targets=(delta/np.linalg.norm(delta,axis=1,keepdims=True)).astype(np.float32)
 anchors=np.asarray([r['student_anchor_zero_based_call'] for r in targets_report['records']],np.int64)
 for i,row in enumerate(targets_report['records']):
  assert row['inventory_index']==i and hashlib.sha256(targets[i].tobytes()).hexdigest()==row['normalized_target_sha256']
 for arm,coefficient in plan['arms'].items():
  path=R/f'4090-external-representation-{arm}-training-original-v1.json';report=json.loads(path.read_text())
  assert report['steps']==report['optimizer_updates']==20000 and report['cached_targets_unchanged'] and report['teacher_inference_calls']==0
  assert report['launch']['plan_sha256']==sha(PLAN) and report['launch']['arm']==arm and report['launch']['coefficient']==coefficient
  assert report['cached_target_sha256']==hashlib.sha256(targets.tobytes()).hexdigest()
  for name,h in report['files'].items():assert sha(ROOT/arm/name)==h
  initial=torch.load(ROOT/arm/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(ROOT/arm/'final.pt',map_location='cpu',weights_only=False)
  assert initial['step']==0 and final['step']==20000 and final['coefficient']==coefficient
  for checkpoint in [initial,final]:
   assert checkpoint['plan_sha256']==sha(PLAN) and checkpoint['arm']==arm and checkpoint['initial_checkpoint_sha256']==sha(plan['initial_checkpoint'])
  from causal_binary40_v1 import Binary40
  native=Binary40();names=list(dict(native.named_parameters()))
  assert list(initial['model'])==list(final['model'])==list(original['model'])==names and len(names)==24
  assert all(torch.equal(initial['model'][k],v) for k,v in original['model'].items())
  assert any(not torch.equal(initial['model'][k],final['model'][k]) for k in names)
  torch.manual_seed(726);projection=torch.nn.Linear(128,768)
  assert list(initial['projection'])==list(final['projection'])==['weight','bias']
  assert all(torch.equal(initial['projection'][k],v) for k,v in projection.state_dict().items())
  assert all(torch.isfinite(v).all() for section in ['model','projection'] for v in final[section].values())
  changed=any(not torch.equal(initial['projection'][k],final['projection'][k]) for k in initial['projection'])
  assert changed==(coefficient==1) and report['control_projection_unchanged']==(coefficient==0)
  groups=final['optimizer']['param_groups'];assert len(groups)==1;group=groups[0]
  assert group['params']==list(range(26)) and group['lr']==.001 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['foreach'] is False
  assert not group['amsgrad'] and not group['maximize']
  states=final['optimizer']['state'];assert set(states)==set(range(26))
  assert all(float(v['step'])==20000 for v in states.values())
  assert all(torch.isfinite(v).all() for state in states.values() for v in state.values() if torch.is_tensor(v))
  if coefficient==0:
   assert all(torch.count_nonzero(states[i][k])==0 for i in [24,25] for k in ['exp_avg','exp_avg_sq'])
  arms[arm]=dict(training_report_sha256=sha(path),final_checkpoint_sha256=sha(ROOT/arm/'final.pt'),optimizer_updates=20000,coefficient=coefficient,projection_changed=changed,optimizer_verified=True)
 with gzip.open(R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz','rt') as f:schedule=[json.loads(x) for x in f]
 count=0;seen=set()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as orig,gzip.open(ROOT/'control/batches.jsonl.gz','rt') as control,gzip.open(ROOT/'feature/batches.jsonl.gz','rt') as decay:
  for step in range(1,20001):
   old=json.loads(next(orig));a=json.loads(next(control));b=json.loads(next(decay));assert old['step']==a['step']==b['step']==step
   for key in ['features_sha256','positive_sha256','negative_sha256']:assert a[key]==b[key]==old[key]
   descriptor=hashlib.sha256(json.dumps(old['descriptors'],sort_keys=True).encode()).hexdigest();assert a['main_descriptor_sha256']==b['main_descriptor_sha256']==descriptor
   assert a['pair_indices']==b['pair_indices']==schedule[step-1]['inventory_indices'];assert a['pair_tensor_hashes']==b['pair_tensor_hashes'] and len(a['pair_tensor_hashes'])==4;seen.update(a['pair_indices'])
   expected_target=hashlib.sha256(np.repeat(targets[a['pair_indices']],2,axis=0).tobytes()).hexdigest()
   expected_anchors=np.repeat(anchors[a['pair_indices']],2).tolist()
   assert a['target_sha256']==b['target_sha256']==expected_target
   assert a['anchor_calls']==b['anchor_calls']==expected_anchors
   for coefficient,row in enumerate([a,b]):
    assert all(np.isfinite(row[k]) for k in ['main_bce','paired_bce','feature_cosine','total'])
    assert np.isclose(row['total'],row['main_bce']+row['paired_bce']+coefficient*row['feature_cosine'],rtol=2e-6,atol=1e-7)
   count+=1
  assert orig.readline()==control.readline()==decay.readline()==''
 assert count==20000 and len(seen)==4320
 out=dict(schema='external-representation-training-review-v1',all_passed=True,every_batch_identical_to_original=True,every_paired_batch_identical_between_arms=True,pair_indices_match_fixed_schedule=True,steps=20000,unique_paired_sources=len(seen),terminal_receipt_sha256=sha(receipt),plan_sha256=sha(PLAN),arms=arms,limits='Exhaustive saved main hashes/descriptors checked against original and paired tensor hashes between arms; no new full tensor generation. All normalized target hashes and anchor calls independently reconstructed; runtime target immutability asserted by original code, not reconstructed intermediate tensors. Recognition/INT8/P4 unproven.')
 (R/'4090-external-representation-training-review-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
