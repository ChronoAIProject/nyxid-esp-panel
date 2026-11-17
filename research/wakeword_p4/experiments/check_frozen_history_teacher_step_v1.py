"""Disposable matched clones: one Adam update each, no retained checkpoint."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-frozen-history-teacher-step-plan-v1.json';CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 p=json.loads((R/'4090-frozen-history-teacher-network-plan-v1.json').read_text());p['files'][__file__]=sha(__file__);f=R/'4090-frozen-history-teacher-network-original-v1.json';p['files'][str(f)]=sha(f);p['policy']='Disposable control0/treatment1 clones at original final weights, first fixed original main batch and first4pair schedule. Both compute same main/pair/KL. Actual backward versus component sum rtol1e-5 atol1e-6; one freshAdam1e-3 step each, all optimizer step counts1, frozen teacher exact unchanged/disconnected. No saved/reused model or formal training. Preserve actual update counts and hashes.';PLAN.write_text(json.dumps(p,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 import torch
 from frozen_history_teacher_pair_v1 import Pairs,tensor,bce,pair_components
 from causal_source_balanced_batch_v1 import Sampler,digest
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load(CP,map_location='cpu',weights_only=False);teacher=Binary40().cuda().eval();teacher.load_state_dict(cp['model'],strict=True);teacher.requires_grad_(False);pairs=Pairs();ids=pairs.schedule[0]['inventory_indices'];batch=pairs.batch(ids);(x,pw,nw),desc=Sampler(726).batch();records=[]
 def statehash(model):
  h=hashlib.sha256()
  for name,v in model.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
  return h.hexdigest()
 teacher_sha=statehash(teacher)
 for coefficient in [0,1]:
  model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True);initial=statehash(model);assert initial==teacher_sha
  optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8);params=tuple(model.parameters());z,_=model(tensor(x).permute(0,2,1),model.initial_state(128,'cuda'));main=bce(z[:,0],tensor(pw),tensor(nw));pair,kl=pair_components(model,teacher,batch);supervised=main+pair
  gs=torch.autograd.grad(supervised,params,retain_graph=True);gd=torch.autograd.grad(kl,params,retain_graph=True);(supervised+coefficient*kl).backward();max_error=0.
  for v,s,d in zip(params,gs,gd):
   expected=s+coefficient*d;assert torch.isfinite(v.grad).all();assert torch.allclose(v.grad,expected,rtol=1e-5,atol=1e-6);max_error=max(max_error,float((v.grad-expected).abs().max()))
  assert all(v.grad is None for v in teacher.parameters());optimizer.step();assert all(float(v['step'])==1 for v in optimizer.state.values());assert statehash(teacher)==teacher_sha
  final=statehash(model);assert final!=initial and all(torch.isfinite(v).all() for v in model.parameters());records.append(dict(coefficient=coefficient,optimizer_updates=1,initial_state_sha256=initial,final_state_sha256=final,main_bce=float(main.detach()),pair_bce=float(pair.detach()),kl=float(kl.detach()),max_gradient_merge_error=max_error,optimizer_parameter_states=len(optimizer.state)))
 assert records[0]['final_state_sha256']!=records[1]['final_state_sha256']
 out=dict(plan_sha256=sha(PLAN),all_passed=True,records=records,teacher_unchanged=True,teacher_gradient_disconnected=True,teacher_state_sha256=teacher_sha,main_descriptor_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),pair_tensor_hashes=[digest(a) for a in batch],saved_model_checkpoints=0,policy=p['policy'])
 (R/'4090-frozen-history-teacher-step-original-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
