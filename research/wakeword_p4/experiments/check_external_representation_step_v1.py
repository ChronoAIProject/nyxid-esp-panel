"""Two disposable actual Adam steps; never reusable as formal training states."""
import gzip,hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-external-representation-step-plan-v1.json';B=Path('.local/wakeword/external-representation-step-v1');CP=Path('.local/wakeword/binary40-training-v1/final.pt')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 review=json.loads((R/'4090-external-representation-network-review-v1.json').read_text());assert review['all_passed'] and review['historical_supervised_gradients_bit_identical']==50
 parent=json.loads((R/'4090-external-representation-network-plan-v1.json').read_text());files=dict(parent['files']);paths=[Path(__file__),R/'4090-external-representation-network-review-v1.json',R/'4090-external-representation-network-original-v1.json',Path('.local/wakeword/external-representation-network-v1/traces.npz')];files.update({str(p):sha(p) for p in paths})
 with LOCK.open('x') as f:json.dump(dict(schema='external-representation-step-plan-v1',files=files,coefficients=[0,1],projection_seed=726,optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0,foreach=False),gradient_tolerance=dict(atol=1e-6,rtol=1e-5),update_tolerance=dict(atol=2e-6,rtol=1e-5),policy='Fresh original student and identical seed726projection per two disposable clones; exactly first original main128/schedule4pair and one Adam update each. Verify actual gradient against saved S+coefficientD and analytic projection derivative; independent first-step Adam formula, all optimizer counters1. Control projection unchanged, treatment updates, fixed targets immutable. Save audit arrays only; no checkpoint/formal reuse, tuning, teacher inference or recognition/P4claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 import torch
 from external_representation_pair_v1 import Targets,components,tensor,bce
 from causal_source_balanced_batch_v1 import Sampler,digest
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load(CP,map_location='cpu',weights_only=False);store=Targets();fixed=store.gpu_targets.clone();(x,pw,nw),desc=Sampler(726).batch()
 with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as f:old=json.loads(next(f))
 assert old['descriptors']==desc
 for key,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]:assert digest(a)==old[key]
 ids=store.pairs.schedule[0]['inventory_indices'];batch=store.batch(ids);B.mkdir(exist_ok=False);arrays={};rows=[]
 with np.load('.local/wakeword/external-representation-network-v1/traces.npz',allow_pickle=False) as z:
  supervised=z['gS_1'].copy();feature=z['gD_1'].copy();q=z['probe_q_1'];target=z['probe_target_1'];ctx=z['probe_context_1'].astype(np.float64);qn=np.linalg.norm(q,axis=1,keepdims=True);tn=np.linalg.norm(target,axis=1,keepdims=True);dq=(-target/(qn*tn)+(q*target).sum(1,keepdims=True)*q/(qn**3*tn))/len(q);projected=np.r_[((dq.T@ctx).astype(np.float32)).ravel(),dq.sum(0).astype(np.float32)]
 for coefficient in [0,1]:
  model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True);torch.manual_seed(726);projection=torch.nn.Linear(128,768).cuda();params=[*model.parameters(),*projection.parameters()];optimizer=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
  initial=np.concatenate([v.detach().cpu().numpy().ravel() for v in params]);projection_initial=[v.detach().clone() for v in projection.parameters()]
  optimizer.zero_grad(set_to_none=True);logits,_=model(tensor(x).permute(0,2,1),model.initial_state(128,'cuda'));main=bce(logits[:,0],tensor(pw),tensor(nw));paired,aux,*_=components(model,projection,batch);loss=main+paired+coefficient*aux;loss.backward()
  assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params);actual=np.concatenate([v.grad.detach().cpu().numpy().ravel() for v in params]);expected=np.r_[supervised+coefficient*feature,coefficient*projected];assert np.allclose(actual,expected,atol=1e-6,rtol=1e-5)
  optimizer.step();updated=np.concatenate([v.detach().cpu().numpy().ravel() for v in params]);reference=(initial.astype(np.float64)-.001*actual.astype(np.float64)/(np.abs(actual.astype(np.float64))+1e-8)).astype(np.float32);assert np.allclose(updated,reference,atol=2e-6,rtol=1e-5) and len(optimizer.state)==len(params) and all(float(v['step'])==1 for v in optimizer.state.values())
  same_projection=all(torch.equal(a,b) for a,b in zip(projection_initial,projection.parameters()));assert same_projection==(coefficient==0);assert torch.equal(fixed,store.gpu_targets)
  arrays.update({f'{coefficient}_initial':initial,f'{coefficient}_actual_gradient':actual,f'{coefficient}_expected_gradient':expected,f'{coefficient}_updated':updated,f'{coefficient}_reference_update':reference})
  rows.append(dict(coefficient=coefficient,main_bce=float(main.detach()),paired_bce=float(paired.detach()),feature_cosine=float(aux.detach()),optimizer_steps=1,optimizer_parameter_states=len(optimizer.state),gradient_max_error=float(np.abs(actual-expected).max()),update_max_error=float(np.abs(updated-reference).max()),control_projection_unchanged=same_projection,student_updated=bool(np.any(updated[:30593]!=initial[:30593]))))
 assert np.array_equal(arrays['0_initial'],arrays['1_initial']) and not np.array_equal(arrays['0_updated'][:30593],arrays['1_updated'][:30593]) and all(r['student_updated'] for r in rows)
 trace=B/'audit.npz';np.savez_compressed(trace,**arrays);out=dict(schema='external-representation-step-original-v1',plan_sha256=sha(LOCK),rows=rows,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),fixed_targets_unchanged=True,formal_training_reuse_eligible=False,checkpoints_saved=False,gpu=torch.cuda.get_device_name(),policy=p['policy'])
 with (R/'4090-external-representation-step-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
