"""One fixed matched pair: supervised physical histories, fixed teacher KL0/1."""
import gzip,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-frozen-history-teacher-training-plan-v1.json';ROOT=Path('.local/wakeword/frozen-history-teacher-training-v1');CP=Path('.local/wakeword/binary40-training-v1/final.pt');ARMS={'control':0,'teacher':1}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def disk():assert shutil.disk_usage('.').free>100*1024**3

def plan():
 from train_binary40_v1 import sources
 old=json.loads((R/'4090-frozen-history-teacher-network-plan-v1.json').read_text());files=dict(old['files']);paths=set();sources(__file__,paths)
 paths.update([R/'4090-frozen-history-teacher-network-review-v1.json',R/'4090-frozen-history-teacher-step-original-v1.json',Path(__file__).with_name('frozen_history_teacher_formal_protocol_v1.md'),Path('research/wakeword_p4/training/gpu_python.py')])
 files.update({str(p):sha(p) for p in paths})
 p=dict(schema='frozen-history-teacher-training-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['torch','numpy']},arms=ARMS,seed=726,steps=20000,optimizer=dict(name='Adam',lr=.001,betas=[.9,.999],eps=1e-8),initial_checkpoint=str(CP),initial_checkpoint_sha256=sha(CP),pair_schedule_seed=237726,progress_every=200,policy='Fresh original Binary40final20000weights for each student and separate permanently frozen teacher. Original main128Sampler726 full20000batches authenticated on the fly against original descriptors/features/masks; fixed same4pairs/step botharms, complete quiet/history forwards and fixed teacher inference in both. Loss=mainBCE+pairedBCE+coefficient*teacherKL, coefficient0/1only difference. FreshAdam, no schedule/clipping/seed/width/threshold/calibration/checkpoint rescue. One update/step, final-only artifacts; no probe clone reuse or scored-source filtering. Both runs sequential, actual subprocess exits recorded. Failure stops supervisor and preserves original artifacts without restart. Export/replay fixed in companion protocol; no P4 or acceptance claim from training.')
 PLAN.write_text(json.dumps(p,indent=2)+'\n')
def verify():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 disk();return p

def run(arm):
 p=verify();coefficient=ARMS[arm];out=ROOT/arm;out.mkdir(parents=True,exist_ok=False)
 launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],arm=arm,coefficient=coefficient,plan_sha256=sha(PLAN));(out/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 try:
  import torch
  from causal_source_balanced_batch_v1 import Sampler,digest
  from frozen_history_teacher_pair_v1 import Pairs,tensor,bce,pair_components
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
  from causal_binary40_v1 import Binary40
  torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(726);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
  assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
  cp=torch.load(CP,map_location='cpu',weights_only=False);assert cp['step']==20000
  model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True);teacher=Binary40().cuda().eval();teacher.load_state_dict(cp['model'],strict=True);teacher.requires_grad_(False)
  optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8);sampler=Sampler(726);pairs=Pairs();start=time.monotonic();progress=[]
  def frozen():return all(torch.equal(v.detach().cpu(),cp['model'][name].detach().cpu()) for name,v in teacher.state_dict().items())
  assert frozen()
  torch.save(dict(step=0,model=model.state_dict(),plan_sha256=sha(PLAN),arm=arm,initial_checkpoint_sha256=sha(CP)),out/'initial.pt')
  with gzip.open('.local/wakeword/binary40-training-v1/batches.jsonl.gz','rt') as original,gzip.open(out/'batches.jsonl.gz','wt') as log:
   for step in range(1,20001):
    (x,pw,nw),desc=sampler.batch();old=json.loads(next(original));assert old['step']==step and old['descriptors']==desc
    hashes={k:digest(a) for k,a in [('features_sha256',x),('positive_sha256',pw),('negative_sha256',nw)]}
    for k,h in hashes.items():assert h==old[k]
    schedule=pairs.schedule[step-1];assert schedule['step']==step;ids=schedule['inventory_indices'];batch=pairs.batch(ids);pair_hashes=[digest(a) for a in batch]
    optimizer.zero_grad(set_to_none=True);z,_=model(tensor(x).permute(0,2,1),model.initial_state(128,'cuda'));main=bce(z[:,0],tensor(pw),tensor(nw));paired,kl=pair_components(model,teacher,batch);loss=main+paired+coefficient*kl
    assert torch.isfinite(loss);loss.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters());assert all(v.grad is None for v in teacher.parameters());optimizer.step();assert all(torch.isfinite(v).all() for v in model.parameters())
    values=dict(main_bce=float(main.detach()),paired_bce=float(paired.detach()),kl=float(kl.detach()),total=float(loss.detach()))
    log.write(json.dumps(dict(step=step,**hashes,main_descriptor_sha256=hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest(),pair_indices=ids,pair_tensor_hashes=pair_hashes,**values))+'\n')
    if step==1 or step%200==0:
     row=dict(step=step,optimizer_updates=step,elapsed_seconds=time.monotonic()-start,**values);progress.append(row);log.flush();(out/'progress.json').write_text(json.dumps(dict(launch=launch,latest=row),indent=2)+'\n');print(json.dumps(row),flush=True)
    if step%2000==0:assert frozen();disk()
   assert original.readline()==''
  assert frozen() and all(float(v['step'])==20000 for v in optimizer.state.values())
  torch.save(dict(step=20000,model=model.state_dict(),optimizer=optimizer.state_dict(),plan_sha256=sha(PLAN),arm=arm,coefficient=coefficient,initial_checkpoint_sha256=sha(CP)),out/'final.pt')
  result=dict(schema='frozen-history-teacher-training-original-v1',launch=launch,steps=20000,optimizer_updates=20000,teacher_unchanged=True,teacher_disconnected=True,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,files={f.name:sha(f) for f in out.iterdir() if f.is_file()},progress=progress,policy=p['policy'])
  (R/f'4090-frozen-history-teacher-{arm}-training-original-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=True,arm=arm,optimizer_updates=20000)),flush=True)
 except Exception:
  (out/'failure.log').write_text(traceback.format_exc());raise

def supervise():
 verify();ROOT.mkdir(exist_ok=False);records=[];status='failed'
 try:
  for arm in ARMS:
   log=ROOT/f'{arm}-execution.log';args=['/usr/bin/python3','research/wakeword_p4/training/gpu_python.py',__file__,'run',arm];start=time.monotonic()
   with log.open('xb') as f:
    child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT);launch=dict(supervisor_pid=os.getpid(),child_pid=child.pid,arm=arm,status='running',args=args,plan_sha256=sha(PLAN));(ROOT/'state.json').write_text(json.dumps(launch,indent=2)+'\n');code=child.wait()
   row=dict(arm=arm,pid=child.pid,actual_exit_code=code,elapsed_seconds=time.monotonic()-start,log_path=str(log),log_sha256=sha(log));records.append(row);print(json.dumps(row),flush=True)
   assert code==0,f'{arm} failed with actual exit {code}'
  status='completed-await-independent-review'
 finally:
  receipt=dict(status=status,plan_sha256=sha(PLAN),stages=records);(ROOT/'state.json').write_text(json.dumps(receipt,indent=2)+'\n');(R/'4090-frozen-history-teacher-training-terminal-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':
 if sys.argv[1]=='run':run(sys.argv[2])
 else:{'plan':plan,'supervise':supervise}[sys.argv[1]]()
