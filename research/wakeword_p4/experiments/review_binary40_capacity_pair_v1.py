"""Review terminal paired artifacts and every batch identity before final export."""
import gzip,json
from pathlib import Path
import torch
from export_binary40_trained_v1 import sha
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/binary40-capacity-pair-v1');LOCK=R/'4090-binary40-capacity-pair-plan-v1.json'

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 receipts=json.loads(Path('/tmp/binary40-capacity-supervisor-receipt-v1.json').read_text())
 assert [r['arm'] for r in receipts]==['control64','wide128'] and all(r['exit_code']==0 for r in receipts)
 plan=json.loads(LOCK.read_text())
 for p,h in plan['files'].items():assert sha(p)==h,p
 reports={};checkpoints={}
 for arm in plan['arms']:
  p=R/f'4090-binary40-capacity-{arm}-training-original-v1.json';r=json.loads(p.read_text());assert r['steps']==20000 and r['arm']==arm and r['plan_sha256']==sha(LOCK)
  assert r['launch']['pid']==next(x['pid'] for x in receipts if x['arm']==arm)
  for name,h in r['files'].items():assert sha(B/arm/name)==h,(arm,name)
  checkpoints[arm]={}
  for name,step in [('initial',0),('final',20000)]:
   cp=torch.load(B/arm/(name+'.pt'),map_location='cpu',weights_only=False);assert cp['step']==step and cp['plan_sha256']==sha(LOCK)
   assert all(torch.isfinite(v).all() for v in cp['model'].values());checkpoints[arm][name]=cp['model']
  reports[arm]=dict(path=str(p),sha256=sha(p),final_sha256=sha(B/arm/'final.pt'))
 parent=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False)['model']
 assert all(torch.equal(v,parent[k]) for k,v in checkpoints['control64']['initial'].items())
 w=checkpoints['wide128']['initial']
 for key in ['stem.weight','stem.bias',*[f'dw.{i}.{suffix}' for i in range(5) for suffix in ['weight','bias']]]:
  assert torch.equal(w[key],parent[key].repeat(2,*([1]*(parent[key].ndim-1))))
 for i in range(5):
  k=f'pw.{i}.weight';assert torch.allclose(w[k][:,:64]+w[k][:,64:],parent[k].repeat(2,1,1),atol=1e-7,rtol=1e-6)
  assert torch.equal(w[f'pw.{i}.bias'],parent[f'pw.{i}.bias'].repeat(2))
 assert torch.allclose(w['head.weight'][:,:64]+w['head.weight'][:,64:],parent['head.weight'],atol=1e-7,rtol=1e-6)
 assert torch.equal(w['head.bias'],parent['head.bias'])
 logs=[Path('.local/wakeword/binary40-training-v1/batches.jsonl.gz'),B/'control64/batches.jsonl.gz',B/'wide128/batches.jsonl.gz'];steps=0
 with gzip.open(logs[0],'rt') as a,gzip.open(logs[1],'rt') as b,gzip.open(logs[2],'rt') as c:
  while True:
   lines=[f.readline() for f in [a,b,c]]
   if not any(lines):break
   assert all(lines);rows=[json.loads(line) for line in lines];steps+=1
   for r in rows:assert r['step']==steps
   for key in ['features_sha256','positive_sha256','negative_sha256','descriptors']:assert rows[0][key]==rows[1][key]==rows[2][key],(steps,key)
 assert steps==20000
 out=dict(schema='binary40-capacity-pair-review-v1',reviewer_sha256=sha(__file__),plan_sha256=sha(LOCK),terminal_receipts=receipts,reports=reports,every_batch_identical_to_original=True,steps_verified=steps,descriptor_rows_verified_per_arm=steps*128,initial_control_exact=True,wide_initial_split_structure_verified=True,all_initial_final_tensors_finite=True,recognition_scored=False,policy='Artifact and sampling audit only. Supervisor exit receipts plus authoritative terminal session must be reported; no recognition/INT8/P4 claim.')
 dest=R/'4090-binary40-capacity-pair-review-v1.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
