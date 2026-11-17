"""Complete fixed final-pair float diagnostic; no optimizer or INT8 inference."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import replay_binary40_float_v2 as legacy
from collect_multispeaker_formal_v1 import prepare
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/shorter-span-float-v1');LOCK=R/'4090-shorter-span-float-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def configure(arm):
 legacy.introot=lambda c:Path(f'.local/wakeword/binary40-shorter-span-{arm}-{c}-replay-v1')
 legacy.intlock=lambda c:R/f'4090-binary40-shorter-span-{arm}-{c}-replay-plan-v1.json'
def plan():
 files={}
 for arm in ['control','shortspan']:
  configure(arm)
  for c in legacy.COHORTS:
   p=legacy.intlock(c);files.update(read(p)['files'])
   for f in [p,legacy.introot(c)/'result.json']:
    files[str(f)]=sha(f)
   for f in legacy.introot(c).rglob('*'):
    if f.is_file():files[str(f)]=sha(f)
  p=R/f'4090-binary40-shorter-span-{arm}-context-plan-v1.json';files.update(read(p)['files'])
  for f in [p,R/f'4090-binary40-shorter-span-{arm}-context-original-v1.json',Path(f'.local/wakeword/binary40-shorter-span-training-v1/{arm}/final.pt')]:files[str(f)]=sha(f)
  for row in read(R/f'4090-binary40-shorter-span-{arm}-context-original-v1.json')['streams']:files[row['trace']['path']]=row['trace']['sha256']
 for folder in ['experiments','training']:
  for f in Path(f'research/wakeword_p4/{folder}').glob('*.py'):files[str(f)]=sha(f)
 p=Path('research/wakeword_p4/experiments/shorter_span_float_diagnostic_protocol_v1.md');files[str(p)]=sha(p)
 with LOCK.open('x') as f:json.dump(dict(schema='shorter-span-float-plan-v1',files=files,arms=['control','shortspan'],cohorts=legacy.COHORTS,contexts=3,chunk_frames=3000,optimizer_updates=0),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=read(LOCK)
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','2'],check=True)
 import torch
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from causal_binary40_span_v1 import Binary40Span,CONTROL,TREATMENT
 from features import features
 from run_parallel_context_streams_v1 import material
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 B.mkdir(exist_ok=False);reports={}
 for arm in p['arms']:
  configure(arm);cp=torch.load(f'.local/wakeword/binary40-shorter-span-training-v1/{arm}/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
  model=Binary40Span(CONTROL if arm=='control' else TREATMENT).cuda().eval();model.load_state_dict(cp['model'],strict=True)
  assert all(torch.equal(v.cpu(),cp['model'][k]) for k,v in model.state_dict().items())
  # Disposable loaded-weight equivalence gate, no updates.
  rng=np.random.default_rng(726);probe=torch.from_numpy(rng.normal(size=(1,40,300)).astype('float32')).cuda()
  with torch.inference_mode():
   full,fs=model(probe,model.initial_state(device='cuda'));state=model.initial_state(device='cuda');parts=[]
   for i in range(0,300,3):
    y,state=model(probe[:,:,i:i+3],state);parts.append(y)
   err=float((full-torch.cat(parts,2)).abs().max());assert err<1e-4,err
   assert all(torch.allclose(a,b,atol=1e-4,rtol=1e-5) for a,b in zip(fs,state))
  root=B/arm;root.mkdir();rows=[];seen=set()
  def process(cohort,index,phase,source,pcm,codes,oldclock,timeline=None):
   key=f'{cohort}-{phase}-{index}';assert key not in seen;seen.add(key)
   whole=np.r_[np.zeros(81600,np.int16),pcm];values,support,_=prepare(features(whole),len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600
   assert np.array_equal(clock,oldclock) and len(codes)==len(values)//3
   state=model.initial_state(device='cuda');parts=[]
   with torch.inference_mode():
    for i in range(0,len(values),3000):
     logits,state=model(torch.from_numpy(values[i:i+3000].T.copy()[None]).cuda(),state);parts.append(torch.sigmoid(logits[0,0]).cpu().numpy())
   prob=np.concatenate(parts);assert np.isfinite(prob).all()
   events={k:[dict(step=int(s),relative_sample=int(clock[s-1])) for s in steps] for k,steps in [('float',legacy.float_events(prob)),('int8',event_steps(codes,.91796875))]}
   path=root/(key+'.npz');np.savez_compressed(path,probability=prob,uint8=codes,relative_block_return=clock)
   row=dict(cohort=cohort,index=index,phase=phase,source=source,events=events,timeline=timeline,trace=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size),pcm_sha256=hashlib.sha256(whole.astype('<i2').tobytes()).hexdigest())
   rows.append(row)
   with (root/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(dict(arm=arm,key=key,calls=len(prob))),flush=True)
  for c in p['cohorts']:
   for index,source,pcm,path,h,q,original,timeline in legacy.inputs(c):
    assert sha(path)==h
    with np.load(path) as z:
     clock=z['block_return_sample']-81600 if c=='melo' else z['real_block_return_sample'] if c in ['primed','jenny','joe'] else z['relative_block_return']
     process(c,index,original['phase'] if original else 'synthetic',source,pcm,z['uint8'].copy(),clock.copy(),timeline)
  streams=read(R/'4090-parallel-context-streams-plan-v1.json')['streams'];saved=read(R/f'4090-binary40-shorter-span-{arm}-context-original-v1.json')['streams']
  for s,ref in zip(streams,saved,strict=True):
   assert s['name']==ref['name'];pcm,timeline=material(s,8000);assert timeline==ref['timeline'];assert sha(ref['trace']['path'])==ref['trace']['sha256']
   with np.load(ref['trace']['path']) as z:process('context',s['name'],'whole',{'name':s['name']},pcm,z['uint8'].copy(),z['relative_block_return'].copy(),timeline)
  assert len(rows)==1763
  path=root/'result.json';path.write_text(json.dumps(dict(arm=arm,rows=rows,loaded_checkpoint_sha256=sha(f'.local/wakeword/binary40-shorter-span-training-v1/{arm}/final.pt'),native_parity_max_error=err,optimizer_updates=0),indent=2)+'\n');reports[arm]=dict(path=str(path),sha256=sha(path),rows=len(rows))
 with (R/'4090-shorter-span-float-original-v1.json').open('x') as f:json.dump(dict(plan_sha256=sha(LOCK),arms=reports,gpu=torch.cuda.get_device_name(),optimizer_updates=0,independent_review_pending=True),f,indent=2);f.write('\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
