"""Terminal/checkpoint and exhaustive original/main/paired provenance review."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/frozen-history-teacher-training-v1');PLAN=R/'4090-frozen-history-teacher-training-plan-v1.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipt=R/'4090-frozen-history-teacher-training-terminal-v1.json';terminal=json.loads(receipt.read_text());assert terminal['status']=='completed-await-independent-review' and terminal['plan_sha256']==sha(PLAN)
 assert [r['arm'] for r in terminal['stages']]==['control','teacher'] and all(r['actual_exit_code']==0 for r in terminal['stages'])
 plan=json.loads(PLAN.read_text())
 for f,h in plan['files'].items():assert sha(f)==h,f
 for row in terminal['stages']:assert sha(row['log_path'])==row['log_sha256']
 original=torch.load(plan['initial_checkpoint'],map_location='cpu',weights_only=False);arms={}
 for arm,coefficient in plan['arms'].items():
  path=R/f'4090-frozen-history-teacher-{arm}-training-original-v1.json';report=json.loads(path.read_text());assert report['steps']==report['optimizer_updates']==20000 and report['teacher_unchanged'] and report['teacher_disconnected'];assert report['launch']['plan_sha256']==sha(PLAN)
  for name,h in report['files'].items():assert sha(ROOT/arm/name)==h
  initial=torch.load(ROOT/arm/'initial.pt',map_location='cpu',weights_only=False);final=torch.load(ROOT/arm/'final.pt',map_location='cpu',weights_only=False)
  assert initial['step']==0 and final['step']==20000 and final['coefficient']==coefficient and final['plan_sha256']==sha(PLAN)
  assert set(initial['model'])==set(final['model'])==set(original['model'])
  assert all(torch.equal(initial['model'][k],v) for k,v in original['model'].items())
  assert all(torch.isfinite(v).all() for v in final['model'].values())
  assert all(float(s['step'])==20000 for s in final['optimizer']['state'].values())
  arms[arm]=dict(training_report_sha256=sha(path),final_checkpoint_sha256=sha(ROOT/arm/'final.pt'),initial_checkpoint_sha256=sha(ROOT/arm/'initial.pt'),teacher_unchanged_reported=True,optimizer_updates=20000)
 with gzip.open(R/'4090-frozen-history-teacher-pair-schedule-v1.jsonl.gz','rt') as f:schedule=[json.loads(x) for x in f]
 count=0;seen=set()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as orig,gzip.open(ROOT/'control/batches.jsonl.gz','rt') as control,gzip.open(ROOT/'teacher/batches.jsonl.gz','rt') as teacher:
  for step in range(1,20001):
   old=json.loads(next(orig));a=json.loads(next(control));b=json.loads(next(teacher));assert old['step']==a['step']==b['step']==step
   for key in ['features_sha256','positive_sha256','negative_sha256']:assert a[key]==b[key]==old[key]
   descriptor=hashlib.sha256(json.dumps(old['descriptors'],sort_keys=True).encode()).hexdigest();assert a['main_descriptor_sha256']==b['main_descriptor_sha256']==descriptor
   assert a['pair_indices']==b['pair_indices']==schedule[step-1]['inventory_indices'];assert a['pair_tensor_hashes']==b['pair_tensor_hashes'] and len(a['pair_tensor_hashes'])==4;seen.update(a['pair_indices'])
   for row in [a,b]:assert all(np.isfinite(row[k]) for k in ['main_bce','paired_bce','kl','total'])
   count+=1
  assert orig.readline()==control.readline()==teacher.readline()==''
 assert count==20000 and len(seen)==4320
 out=dict(schema='frozen-history-teacher-training-review-v1',all_passed=True,every_batch_identical_to_original=True,every_paired_batch_identical_between_arms=True,pair_indices_match_fixed_schedule=True,steps=20000,unique_paired_sources=len(seen),terminal_receipt_sha256=sha(receipt),plan_sha256=sha(PLAN),arms=arms,limits='Exhaustive saved main hashes/descriptors checked against original and paired tensor hashes between arms; no new full tensor generation. Teacher immutability enforced in original training code and reported, not reconstructed intermediate tensors. Recognition/INT8/P4 unproven.')
 (R/'4090-frozen-history-teacher-training-review-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
