"""Review terminal paired artifacts and every batch identity before final export."""
import gzip,json
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-multiscale-training-v1');LOCK=R/'4090-binary40-multiscale-training-plan-v1.json'

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipts=json.loads(Path('research/wakeword_p4/results/4090-binary40-multiscale-training-receipts-v1.json').read_text())
 assert [r['arm'] for r in receipts]==['control','multiscale'] and all(r['exit_code']==0 for r in receipts)
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 reports={};checkpoints={}
 for arm in plan['arms']:
  p=R/f'4090-binary40-multiscale-{arm}-training-original-v1.json';r=json.loads(p.read_text());assert r['steps']==20000 and r['arm']==arm and r['plan_sha256']==sha(LOCK)
  assert r['launch']['pid']==next(x['pid'] for x in receipts if x['arm']==arm)
  for name,h in r['files'].items():assert sha(B/arm/name)==h,(arm,name)
  checkpoints[arm]={}
  for name,step in [('initial',0),('final',20000)]:
   cp=torch.load(B/arm/(name+'.pt'),map_location='cpu',weights_only=False);assert cp['step']==step and cp['plan_sha256']==sha(LOCK)
   if step==0:assert not cp['optimizer']['state']
   assert all(torch.isfinite(v).all() for v in cp['model'].values());checkpoints[arm][name]=cp['model']
  reports[arm]=dict(path=str(p),sha256=sha(p),final_sha256=sha(B/arm/'final.pt'))
 parent=torch.load('.local/wakeword/binary40-training-v1/initial.pt',map_location='cpu',weights_only=False)['model']
 assert all(torch.equal(v,parent[k]) for k,v in checkpoints['control']['initial'].items())
 ci=checkpoints['control']['initial'];mi=checkpoints['multiscale']['initial']
 assert all(torch.equal(v,mi[k]) for k,v in ci.items() if not k.startswith('head.'))
 assert torch.count_nonzero(mi['head.weight'][:,:320,:])==0
 assert torch.equal(mi['head.weight'][:,320:,:],ci['head.weight']) and torch.equal(mi['head.bias'],ci['head.bias'])
 for arm in plan['arms']:
  final=torch.load(B/arm/'final.pt',map_location='cpu',weights_only=False);assert final['arm']==arm and final['geometry']['parameters']==(30593 if arm=='control' else 31233);opt=final['optimizer'];assert opt['state'] and all(int(v['step'])==20000 for v in opt['state'].values())
  assert all(torch.isfinite(v[k]).all() for v in opt['state'].values() for k in ['exp_avg','exp_avg_sq'])
 logs=[Path('.local/wakeword/binary40-training-v1/batches.jsonl.gz'),B/'control/batches.jsonl.gz',B/'multiscale/batches.jsonl.gz'];steps=0
 with gzip.open(logs[0],'rt') as a,gzip.open(logs[1],'rt') as b,gzip.open(logs[2],'rt') as c:
  while True:
   lines=[f.readline() for f in [a,b,c]]
   if not any(lines):break
   assert all(lines);rows=[json.loads(line) for line in lines];steps+=1
   for r in rows:assert r['step']==steps
   for key in ['features_sha256','positive_sha256','negative_sha256','descriptors']:assert rows[0][key]==rows[1][key]==rows[2][key],(steps,key)
 assert steps==20000
 out=dict(schema='binary40-multiscale-training-review-v1',reviewer_sha256=sha(__file__),plan_sha256=sha(LOCK),terminal_receipts=receipts,reports=reports,every_batch_identical_to_original=True,steps_verified=steps,descriptor_rows_verified_per_arm=steps*128,initial_control_exact=True,same_initial_backbone_tensors_exact=True,function_preserving_head_initialization_verified=True,all_initial_final_tensors_finite=True,recognition_scored=False,policy='Artifact and sampling audit only. Supervisor exit receipts plus authoritative terminal session must be reported; no recognition/INT8/P4 claim.')
 dest=R/'4090-binary40-multiscale-training-review-v1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
