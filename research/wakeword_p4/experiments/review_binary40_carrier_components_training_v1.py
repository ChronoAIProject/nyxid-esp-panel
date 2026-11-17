"""Review terminal paired artifacts and every batch identity before final export."""
import gzip,json
import numpy as np
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-carrier-components-training-v1');LOCK=R/'4090-binary40-carrier-components-training-plan-v1.json'

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipts=json.loads(Path('.local/wakeword/carrier-components-supervisor-v1/receipts.json').read_text())
 assert [r['arm'] for r in receipts]==['positive_only','negative_only'] and all(r['actual_exit_code']==0 for r in receipts)
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 reports={};checkpoints={}
 for arm in plan['arms']:
  p=R/f'4090-binary40-carrier-components-{arm}-training-original-v1.json';r=json.loads(p.read_text());assert r['steps']==20000 and r['arm']==arm and r['plan_sha256']==sha(LOCK)
  assert r['launch']['pid']==next(x['pid'] for x in receipts if x['arm']==arm)
  for name,h in r['files'].items():assert sha(B/arm/name)==h,(arm,name)
  checkpoints[arm]={}
  for name,step in [('initial',0),('final',20000)]:
   cp=torch.load(B/arm/(name+'.pt'),map_location='cpu',weights_only=False);assert cp['step']==step and cp['plan_sha256']==sha(LOCK)
   if step==0:assert not cp['optimizer']['state']
   assert all(torch.isfinite(v).all() for v in cp['model'].values());checkpoints[arm][name]=cp['model']
  reports[arm]=dict(path=str(p),sha256=sha(p),final_sha256=sha(B/arm/'final.pt'))
 parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
 assert all(torch.equal(v,parent[k]) for k,v in checkpoints['positive_only']['initial'].items())
 ci=checkpoints['positive_only']['initial'];mi=checkpoints['negative_only']['initial']
 assert ci.keys()==mi.keys() and all(torch.equal(v,mi[k]) for k,v in ci.items())
 for arm in plan['arms']:
  final=torch.load(B/arm/'final.pt',map_location='cpu',weights_only=False);assert final['arm']==arm and final['geometry']['parameters']==30593;opt=final['optimizer'];assert opt['state'] and all(int(v['step'])==20000 for v in opt['state'].values())
  assert all(torch.isfinite(v[k]).all() for v in opt['state'].values() for k in ['exp_avg','exp_avg_sq'])
 from causal_source_balanced_batch_v1 import Sampler as OriginalSampler,digest
 from causal_paired_carrier_batch_v1 import Sampler as PairedSampler
 from paired_carrier_batch_audit_v1 import audit
 original_sampler=OriginalSampler(726);paired_sampler=PairedSampler(726)
 logs=[Path('.local/wakeword/binary40-training-v1/batches.jsonl.gz'),B/'positive_only/batches.jsonl.gz',B/'negative_only/batches.jsonl.gz'];steps=0
 with gzip.open(logs[0],'rt') as a,gzip.open(logs[1],'rt') as b,gzip.open(logs[2],'rt') as c:
  while True:
   lines=[f.readline() for f in [a,b,c]]
   if not any(lines):break
   assert all(lines);rows=[json.loads(line) for line in lines];steps+=1
   for r in rows:assert r['step']==steps
   old,od=original_sampler.batch();replacement,rd=paired_sampler.batch()
   audit(old,od,replacement,rd)
   assert rows[0]['descriptors']==od
   for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],old,strict=True):assert rows[0][key]==digest(value)
   for row,kind in zip(rows[1:],['paired_carrier_positive','paired_carrier_nick_negative'],strict=True):
    slots=[i for i,r in enumerate(rd) if r['kind']==kind];assert len(slots)==4
    new=tuple(np.concatenate([v,w[slots]],axis=0) for v,w in zip(old,replacement,strict=True));nd=od+[rd[i] for i in slots]
    assert row['batch_audit']==dict(original128_exact=True,appended_rows=4) and row['descriptors']==nd
    assert all(np.array_equal(v,w[:128]) for v,w in zip(old,new,strict=True))
    for key,value in zip(['features_sha256','positive_sha256','negative_sha256'],new,strict=True):assert row[key]==digest(value)
 assert steps==20000
 out=dict(schema='binary40-carrier-components-training-review-v1',reviewer_sha256=sha(__file__),plan_sha256=sha(LOCK),terminal_receipts=receipts,reports=reports,all_sampling_checks_passed=True,original128_preserved_both_arms=True,treatment_every_batch_independently_reconstructed=True,steps_verified=steps,descriptor_rows_verified_per_arm=dict(positive_only=steps*132,negative_only=steps*132),initial_original_exact=True,same_initial_backbone_tensors_exact=True,both_initial_models_exact=True,all_initial_final_tensors_finite=True,recognition_scored=False,policy='Artifact and sampling audit only. Supervisor exit receipts plus authoritative terminal session must be reported; no recognition/INT8/P4 claim.')
 dest=R/'4090-binary40-carrier-components-training-review-v1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
