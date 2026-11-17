"""One fresh matched scalar control versus fixed external acoustic target loss."""
import gzip,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-external-representation-training-plan-v1.json';ROOT=Path('.local/wakeword/external-representation-training-v1');CP=Path('.local/wakeword/binary40-training-v1/final.pt');ARMS={'control':0,'feature':1}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 from train_binary40_v1 import sources
 review=json.loads((R/'4090-external-representation-step-review-v1.json').read_text());assert review['all_passed'] and not review['formal_reuse_eligible']
 files=dict(json.loads((R/'4090-external-representation-step-plan-v1.json').read_text())['files']);paths=set();sources(__file__,paths);paths.update([R/'4090-external-representation-step-original-v1.json',R/'4090-external-representation-step-review-v1.json',R/'4090-external-representation-step-execution-v1.json',Path(__file__).with_name('external_representation_formal_protocol_v1.md'),Path('research/wakeword_p4/training/gpu_python.py')]);files.update({str(p):sha(p) for p in paths})
 p=dict(schema='external-representation-training-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['torch','numpy']},arms=ARMS,seed=726,projection_seed=726,steps=20000,optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=0,foreach=False),initial_checkpoint=str(CP),initial_checkpoint_sha256=sha(CP),pair_schedule_seed=237726,policy='Fresh original student/seed726projection per arm, original20000main128batches and same4physical pair schedule, fixed causal external targets. Loss mainBCE+pairBCE+coefficientCosine, coefficients0/1only difference. FreshAdamforeachFalse; sequential20000updates finalonly. No probe/historical control reuse, teacher inference, data/coef/layer/threshold/calibration/checkpoint rescue or automatic promotion. Full final INT8/eight consumed cohorts protocol separately bound.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 assert shutil.disk_usage('.').free>100*1024**3;return p

def run(arm):
 p=verify();coefficient=ARMS[arm];out=ROOT/arm;out.mkdir(parents=True,exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],arm=arm,coefficient=coefficient,plan_sha256=sha(PLAN));(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 try:
  import torch
  from external_representation_pair_v1 import Targets,components,tensor,bce
  from causal_source_balanced_batch_v1 import Sampler,digest
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
  from causal_binary40_v1 import Binary40
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
  cp=torch.load(CP,map_location='cpu',weights_only=False);assert cp['step']==20000;model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True);torch.manual_seed(726);projection=torch.nn.Linear(128,768).cuda();initial_projection={k:v.detach().clone() for k,v in projection.state_dict().items()}
  params=[*model.parameters(),*projection.parameters()];optimizer=torch.optim.Adam(params,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False);sampler=Sampler(726);store=Targets();initial_target_hash=digest(store.targets);fixed_gpu_targets=store.gpu_targets.clone();progress=[];started=time.monotonic()
  assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
  torch.save(dict(step=0,model=model.state_dict(),projection=projection.state_dict(),plan_sha256=sha(PLAN),arm=arm,initial_checkpoint_sha256=sha(CP)),out/'initial.pt')
  def immutable():
   assert digest(store.targets)==initial_target_hash and torch.equal(store.gpu_targets,fixed_gpu_targets)
   if coefficient==0:assert all(torch.equal(initial_projection[k],v) for k,v in projection.state_dict().items())
  with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as original,gzip.open(out/'batches.jsonl.gz','wt') as ledger:
   for step in range(1,20001):
    (x,pw,nw),desc=sampler.batch();old=json.loads(next(original));assert old['step']==step and old['descriptors']==desc
    hashes={key:digest(a) for key,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]}
    for key,h in hashes.items():assert h==old[key]
    schedule=store.pairs.schedule[step-1];assert schedule['step']==step;ids=schedule['inventory_indices'];batch=store.batch(ids)
    optimizer.zero_grad(set_to_none=True);z,_=model(tensor(x).permute(0,2,1),model.initial_state(128,'cuda'));main=bce(z[:,0],tensor(pw),tensor(nw));paired,feature,_,_,_,target=components(model,projection,batch);assert not target.requires_grad;loss=main+paired+coefficient*feature;assert torch.isfinite(loss);loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
    if coefficient==0:assert all(torch.count_nonzero(v.grad)==0 for v in projection.parameters())
    optimizer.step();assert all(torch.isfinite(v).all() for v in params)
    values=dict(main_bce=float(main.detach()),paired_bce=float(paired.detach()),feature_cosine=float(feature.detach()),total=float(loss.detach()))
    ledger.write(json.dumps(dict(step=step,**hashes,main_descriptor_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),pair_indices=ids,pair_tensor_hashes=[digest(a) for a in batch[0]],target_sha256=digest(batch[1]),anchor_calls=batch[2].tolist(),**values))+'\n')
    if step==1 or step%200==0:
     row=dict(step=step,optimizer_updates=step,elapsed_seconds=time.monotonic()-started,**values);progress.append(row);ledger.flush();(out/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
    if step%2000==0:immutable();assert shutil.disk_usage('.').free>100*1024**3
   assert original.readline()==''
  immutable();assert len(optimizer.state)==26 and all(float(v['step'])==20000 for v in optimizer.state.values())
  torch.save(dict(step=20000,model=model.state_dict(),projection=projection.state_dict(),optimizer=optimizer.state_dict(),plan_sha256=sha(PLAN),arm=arm,coefficient=coefficient,initial_checkpoint_sha256=sha(CP)),out/'final.pt')
  result=dict(schema='external-representation-training-original-v1',launch=launch,steps=20000,optimizer_updates=20000,cached_targets_unchanged=True,cached_target_sha256=initial_target_hash,control_projection_unchanged=all(torch.equal(initial_projection[k],v) for k,v in projection.state_dict().items()),teacher_inference_calls=0,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-started,files={f.name:sha(f) for f in out.iterdir() if f.is_file()},progress=progress,policy=p['policy'])
  with (R/f'4090-external-representation-{arm}-training-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
  print(json.dumps(dict(completed=True,arm=arm,optimizer_updates=20000)),flush=True)
 except Exception:
  (out/'failure.log').write_text(traceback.format_exc());raise

def supervise():
 verify();ROOT.mkdir(exist_ok=False);records=[];status='failed-preserved';identity=dict(supervisor_pid=os.getpid(),supervisor_starttime=Path('/proc/self/stat').read_text().split()[21])
 try:
  for arm in ARMS:
   path=ROOT/f'{arm}-execution.log';args=['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py',__file__,'run',arm];started=time.monotonic()
   with path.open('x') as log:
    child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);(ROOT/'state.json').write_text(json.dumps(dict(**identity,status='running',child_pid=child.pid,arm=arm,args=args,plan_sha256=sha(PLAN)),indent=2)+'\n');rc=child.wait()
   row=dict(arm=arm,pid=child.pid,actual_exit_code=rc,elapsed_seconds=time.monotonic()-started,log_path=str(path),log_sha256=sha(path));records.append(row);print(json.dumps(row),flush=True);assert rc==0,row
  status='completed-await-independent-review'
 finally:
  terminal=dict(**identity,status=status,plan_sha256=sha(PLAN),stages=records);(ROOT/'state.json').write_text(json.dumps(terminal,indent=2)+'\n');(R/'4090-external-representation-training-terminal-v1.json').write_text(json.dumps(terminal,indent=2)+'\n')
 if status!='completed-await-independent-review':raise SystemExit(1)
if __name__=='__main__':
 if sys.argv[1]=='run':run(sys.argv[2])
 else:{'plan':plan,'supervise':supervise}[sys.argv[1]]()
