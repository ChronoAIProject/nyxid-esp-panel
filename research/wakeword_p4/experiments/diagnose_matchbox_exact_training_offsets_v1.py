"""Frozen INT8 scores at exact existing training offsets; no model comparison gate."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import importlib.metadata as metadata
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');LOCK=R/'4090-matchbox-exact-training-offset-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 parent=R/'4090-matchbox-long-training-stream-plan-v1.json';p=json.loads(parent.read_text());p.update(parent_plan_sha256=sha(parent),offsets=[256,512,1024],policy='Training-only exact endpoint diagnosis using the same96 already consumed recordings and same two frozenINT8models. Full44800sample pastPCM windows at each original training endpoint256/512/1024, no crop of generated source or fullRIR. CUDA savedMFCC, CPU singlethread actualINT8, no FP32 recognition. Not streaming event evaluation: three isolated scalar outputs cannot be treated as five-call support. No retraining, threshold selection, qualification, new source or acceptance access. Preserve actualUINT8codes and input hashes. <1GiB disk, CUDA batch32, one CPU thread.')
 p['files'][str(Path(__file__))]=sha(__file__);p['files'][str(parent)]=sha(parent)
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(sha(LOCK))
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
 import numpy as np
 import soundfile as sf
 import torch
 from ai_edge_litert.interpreter import Interpreter
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from matchboxnet_long_window_v1 import LongLogMFCC
 from matchboxnet_pretrained_v1 import FRONTEND_KEYS
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;assert torch.cuda.is_available();results=[]
 for arm,model in p['models'].items():
  state=torch.load(B/f"matchbox-long-{arm}-{'resumed' if arm=='control' else 'recovery'}-v1/frontend.pt",map_location='cpu',weights_only=True)
  front=LongLogMFCC({k:state[n] for k,n in zip(FRONTEND_KEYS,['dct','window','mel'],strict=True)}).cuda().eval()
  it=Interpreter(model_path=model['path'],num_threads=1);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0];scale,zero=a['quantization']
  assert a['dtype']==np.int8 and b['dtype']==np.uint8 and list(a['shape'])==[1,280,1,64] and b['quantization']==(1/256,0)
  for index,row in enumerate(p['records']):
   assert sha(row['path'])==row['sha256'];pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'];assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
   end=row['source_lead']+row['complete_source_and_rir_samples'];windows=[]
   for offset in p['offsets']:
    start=end+offset-44800;assert 0<=start<=row['source_lead'] and end+offset<=len(pcm);windows.append(pcm[start:end+offset])
   raw=np.ascontiguousarray(windows,dtype=np.float32)/32768
   with torch.no_grad():values=front(torch.from_numpy(raw).cuda()).permute(0,2,1).unsqueeze(2).cpu().numpy()
   assert np.isfinite(values).all();q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);codes=[]
   for j in range(3):
    it.set_tensor(a['index'],q[j:j+1]);it.invoke();codes.append(int(it.get_tensor(b['index'])[0,0]))
   results.append(dict(arm=arm,index=index,engine=row['engine'],conditioning=row['conditioning'],class_id=row['class_id'],path=row['path'],codes=codes,quantized_input_sha256=hashlib.sha256(q.tobytes()).hexdigest()))
 out=dict(schema='matchbox-exact-training-offset-original-v1',plan_sha256=sha(LOCK),model_sha256={a:m['sha256'] for a,m in p['models'].items()},records=results,calls=576,all_complete=True,policy=p['policy'])
 with (R/'4090-matchbox-exact-training-offset-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 for x in results:
  if x['arm']=='pooled' and x['class_id']==1:print(json.dumps(x))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
