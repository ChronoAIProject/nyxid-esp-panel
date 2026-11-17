"""Training-only 2x2 frontend-history/cache diagnostic; no optimizer."""
import hashlib,json,sys,subprocess,time
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');PLAN=R/'4090-binary40-context-factorial-plan-v1.json';OUT=Path('.local/wakeword/binary40-context-factorial-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 material=Path('.local/wakeword/causal-onset-material-v1/records.jsonl');rows=[json.loads(x) for x in material.read_text().splitlines()]
 positives=[];seen=set()
 for r in rows:
  s=r['source'];assert s['role']=='training' and r['source_role']=='training'
  if s['label']==1 and s['id'] not in seen:positives.append(s);seen.add(s['id'])
  if len(positives)==12:break
 negative=next(r['source'] for r in rows if r['source']['label']==0)
 files=[Path(__file__),material,Path('.local/wakeword/binary40-training-v1/final.pt'),Path('research/wakeword_p4/training/features.py'),Path('research/wakeword_p4/training/causal_binary40_v1.py')]
 for s in positives+[negative]:assert sha(s['native_path'])==s['native_sha256'];files.append(Path(s['native_path']))
 p=dict(schema='binary40-context-factorial-plan-v1',positives=positives,negative=negative,files={str(f):sha(f) for f in files},gap_samples=8000,prime_samples=81600,tail_samples=16000,policy='First12 unique positive native sources in original training ledger order, first negative native source. No selection by score. Speech-prefix versus equal-length zeros; negative source padded to480samples, then8000gap padded to480. Entire prefix has multiple480samples. Split at complete3feature call before target boundary; leftover frontend frame retained in suffix. Cross speech/silence frontend suffix with speech/silence neural prefix cache. All four final20000 GPUfloat arms, strict weights, no updates. Physical diagonal must reproduce unsplit full inference. Off-diagonals are artificial interventions, not deployable reset behavior. Preserve complete arrays/identities, fixed five-mean threshold, no dev/acceptance audio, no parameter search or qualification.')
 PLAN.write_text(json.dumps(p,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 OUT.mkdir(exist_ok=False)
 import torch,soundfile as sf
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from features import features
 from causal_binary40_v1 import Binary40
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 cp=torch.load('.local/wakeword/binary40-training-v1/final.pt',map_location='cpu',weights_only=False);assert cp['step']==20000
 model=Binary40().cuda().eval();model.load_state_dict(cp['model'],strict=True)
 negative,rate=sf.read(p['negative']['native_path'],dtype='int16');assert rate==16000 and negative.ndim==1
 prefix=np.r_[np.zeros(81600,np.int16),negative,np.zeros((-len(negative))%480+8160,np.int16)].astype(np.int16);assert len(prefix)%480==0
 records=[];arrays={};start=time.monotonic()
 with torch.no_grad():
  for i,source in enumerate(p['positives']):
   target,rate=sf.read(source['native_path'],dtype='int16');assert rate==16000 and target.ndim==1
   xs={};states={};full={};split=None
   for history in ['silence','speech']:
    pre=np.zeros_like(prefix) if history=='silence' else prefix
    raw=features(np.r_[pre,target,np.zeros(16000,np.int16)])
    x=np.clip(raw.astype(np.float32)/np.float32(25.6),0,26);x=x[:len(x)//3*3]
    k=((len(prefix)//160-2)//3)*3
    assert k>0 and k%3==0
    split=k;xt=torch.from_numpy(x.T.copy()[None]).cuda();xs[history]=xt
    before,st=model(xt[:,:,:k],model.initial_state(1,'cuda'));states[history]=st
    f,_=model(xt,model.initial_state(1,'cuda'));full[history]=torch.sigmoid(f[0,0]).cpu().numpy()
    arrays[f'{i}_{history}_features']=x
   detail={}
   for front in ['silence','speech']:
    for cache in ['silence','speech']:
     logits,_=model(xs[front][:,:,split:],states[cache]);prob=torch.sigmoid(logits[0,0]).cpu().numpy()
     if front==cache:assert np.allclose(prob,full[front][split//3:],atol=1e-6,rtol=1e-6)
     arrays[f'{i}_{front}_{cache}_probability']=prob
     # Support formula is the existing original mask ledger, not phonetic truth.
     support=(3*np.arange(split//3+1,split//3+len(prob)+1)+2)*160
     valid=(support[4:]>=len(prefix))&(support[4:]<=len(prefix)+len(target)+4320)
     five=np.convolve(prob.astype(np.float64),np.ones(5)/5,'valid');assert valid.any()
     detail[f'{front}/{cache}']=dict(max_five=float(five[valid].max()),crossing=bool(np.any(five[valid]>=.91796875)),diagonal_max_error=float(np.abs(prob-full[front][split//3:]).max()) if front==cache else None)
   records.append(dict(index=i,source=source,prefix_samples=len(prefix),split_feature=split,arms=detail));print(json.dumps(records[-1]),flush=True)
 trace=OUT/'arrays.npz';np.savez_compressed(trace,**arrays)
 result=dict(schema='binary40-context-factorial-original-v1',plan_sha256=sha(PLAN),records=records,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,optimizer_updates=0,policy=p['policy'])
 (R/'4090-binary40-context-factorial-original-v1.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
