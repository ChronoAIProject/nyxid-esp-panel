"""Bounded training-only full-recording INT8 diagnosis, never qualification."""
import argparse,hashlib,json,os,subprocess,sys,time
import importlib.metadata as metadata
import importlib.util
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
LOCK=R/'4090-matchbox-long-training-stream-plan-v1.json'
OUT=B/'matchbox-long-training-stream-v1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 rows=json.loads((B/'matchbox-long-material-v1/records.json').read_text());selected=[]
 for engine in ['speecht5','piper','kokoro','qwen']:
  candidates=[r for r in rows if r.get('engine')==engine]
  conditions=sorted({r['conditioning'] for r in candidates},key=lambda s:hashlib.sha256(('training-stream-v1/'+s).encode()).hexdigest())[:4]
  for condition in conditions:
   groups={}
   for r in candidates:
    if r['conditioning']==condition:groups.setdefault((r['scale'],r['variant']),[]).append(r)
   group=min(groups,key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())
   for label in range(1,7):
    row=min((r for r in groups[group] if r['class_id']==label),key=lambda r:hashlib.sha256(r['path'].encode()).hexdigest())
    assert row['role']=='training' and sha(row['path'])==row['sha256'];selected.append(row)
 assert len(selected)==96
 files=[Path(__file__),B/'matchbox-long-material-v1/records.json',R/'4090-matchbox-long-training-material-v1.json']
 for name in ['finite_pcm_long_window_v1.py','uint8_event_policy_v1.py']:files.append(Path(__file__).with_name(name))
 for name in ['matchboxnet_long_window_v1.py','matchboxnet_pretrained_v1.py']:files.append(Path('research/wakeword_p4/training')/name)
 spec=importlib.util.find_spec("ai_edge_litert");files.extend(f for f in Path(spec.origin).parent.rglob("*") if f.is_file() and f.suffix in [".py",".so"])
 import torch
 files.extend([Path(torch.__file__),Path(torch._C.__file__),Path(torch.__file__).parent/"lib/libtorch_cuda.so"])
 models={}
 for arm in ['control','pooled']:
  report=R/f'4090-matchbox-long-{arm}-int8-original-v1.json';model=json.loads(report.read_text())['model'];models[arm]=model
  files.extend([report,Path(model['path']),B/f"matchbox-long-{arm}-{'resumed' if arm=='control' else 'recovery'}-v1/frontend.pt"])
 p=dict(versions={n:metadata.version(n) for n in ["torch","numpy","soundfile","ai-edge-litert"]},records=selected,models=models,files={str(f):sha(f) for f in files},threshold=.91796875,prime_samples=81600,policy='Training-only diagnosis, not independent validation. Deterministic hash-ranked four conditioning IDs per engine, one shared scale/variant group each, all six classes; one hash-ranked native record per class (Qwen multiplicity not reweighted). Full original recordings, no cuts, same continuous zero prime/cadence/finish/event policy. CUDA saved MFCC frontend, CPU singlethread actual frozen INT8. No training, threshold selection, qualification, fresh data, phonetic truth or P4 performance claim. Preserve every trace and event. Interpretation: source fitting and nearword confusion only; commands absent from training material, cannot infer command fit. Resource: one CPU thread, CUDA batch32, <1GiB new disk; no PCM duplication.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(sha(LOCK),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
 p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','1'],check=True)
 import numpy as np
 import soundfile as sf
 import torch
 from ai_edge_litert.interpreter import Interpreter
 from finite_pcm_long_window_v1 import contexts
 from uint8_event_policy_v1 import event_steps
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
 from matchboxnet_long_window_v1 import LongLogMFCC
 from matchboxnet_pretrained_v1 import FRONTEND_KEYS
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available();OUT.mkdir(exist_ok=False);start=time.monotonic()
 (OUT/'launch.json').write_text(json.dumps(dict(pid=os.getpid(),plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name())))
 summary={}
 for arm,model in p['models'].items():
  state=torch.load(B/f"matchbox-long-{arm}-{'resumed' if arm=='control' else 'recovery'}-v1/frontend.pt",map_location='cpu',weights_only=True)
  front=LongLogMFCC({k:state[n] for k,n in zip(FRONTEND_KEYS,['dct','window','mel'],strict=True)}).cuda().eval();counts={}
  with (OUT/f'{arm}.jsonl').open('x') as log:
   for index,row in enumerate(p['records']):
    assert sha(row['path'])==row['sha256'];pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples']
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
    windows,nominal,actual,finish=contexts(np.concatenate([np.zeros(p['prime_samples'],np.int16),pcm]));codes=np.empty(len(windows),np.uint8)
    it=Interpreter(model_path=model['path'],num_threads=1);it.allocate_tensors();a=it.get_input_details()[0];b=it.get_output_details()[0]
    assert a['dtype']==np.int8 and b['dtype']==np.uint8 and list(a['shape'])==[1,280,1,64] and b['quantization']==(1/256,0)
    scale,zero=a['quantization'];qh=hashlib.sha256()
    for offset in range(0,len(windows),32):
     raw=np.ascontiguousarray(windows[offset:offset+32],dtype=np.float32)/32768
     with torch.no_grad():values=front(torch.from_numpy(raw).cuda()).permute(0,2,1).unsqueeze(2).cpu().numpy()
     assert np.isfinite(values).all();q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8);qh.update(q.tobytes())
     for j in range(len(q)):
      it.set_tensor(a['index'],q[j:j+1]);it.invoke();codes[offset+j]=it.get_tensor(b['index'])[0,0]
    trace=OUT/f'{arm}-{index:03}.npz';np.savez_compressed(trace,uint8=codes,nominal_sample_end=nominal,actual_sample_end=actual,finish_call=finish)
    events=[dict(step=int(s),sample=int(actual[s-1])-p['prime_samples'],finish=bool(finish[s-1])) for s in event_steps(codes,p['threshold'])]
    key=f"{row['engine']}/class{row['class_id']}";c=counts.setdefault(key,dict(clips=0,activated=0,prime_events=0,leading_events=0))
    c['clips']+=1;c['activated']+=int(any(e['sample']>=row['source_lead'] for e in events));c['prime_events']+=sum(e['sample']<=0 for e in events);c['leading_events']+=sum(0<e['sample']<row['source_lead'] for e in events)
    receipt=dict(index=index,record=row,model_sha256=model['sha256'],events=events,trace=str(trace),trace_sha256=sha(trace),quantized_input_sha256=qh.hexdigest())
    log.write(json.dumps(receipt)+'\n');log.flush();print(json.dumps(dict(arm=arm,completed=index+1,total=96,elapsed=time.monotonic()-start)),flush=True)
  summary[arm]=dict(counts=counts,records_sha256=sha(OUT/f'{arm}.jsonl'))
 result=dict(schema='matchbox-long-training-stream-v1',plan_sha256=sha(LOCK),summary=summary,complete=True,elapsed_seconds=time.monotonic()-start,policy=p['policy'])
 with (R/'4090-matchbox-long-training-stream-original-v1.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
