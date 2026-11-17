"""Actual interleaved two-interpreter execution and online fixed event arbitration."""
import hashlib,json,os,subprocess,sys,time,traceback,shutil,importlib.metadata
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
from c161_binary40_parallel_sources_v1 import inputs
from replay_c161_binary40_parallel_events_v2 import sha,read,COHORTS
from collect_multispeaker_formal_v1 import prepare
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/c161-binary40-parallel-runtime-v1');LOCK=R/'4090-c161-binary40-parallel-runtime-plan-v1.json'
MODELS=dict(c161='.local/wakeword/causal-source-balanced-int8-v1/int8.tflite',binary40='.local/wakeword/binary40-trained-int8-v1/int8.tflite')
class Events:
 def __init__(self):self.history=[];self.last=-1000;self.armed=True
 def update(self,step,code):
  self.history.append(int(code))
  if len(self.history)>5:self.history.pop(0)
  if len(self.history)<5:return False
  total=sum(self.history)
  if total*2<1175 and step-self.last>=67:self.armed=True
  if self.armed and total>=1175 and step-self.last>=67:self.last=step;self.armed=False;return True
  return False

def dual(values,clock):
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 def fresh(path):
  it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors();return it
 a=fresh(MODELS['c161']);b=fresh(MODELS['binary40']);ai=a.get_input_details()[0];ao=a.get_output_details()[0];runner=b.get_signature_runner();bi=runner.get_input_details();bo=runner.get_output_details()
 assert ai['dtype']==np.int8 and ao['dtype']==np.uint8 and ao['quantization']==(1/256,0)
 assert list(ai['shape'])==[1,3,40] and bo['probability']['quantization']==(1/256,-128)
 q={}
 for name,detail in [('c161',ai),('binary40',bi['chunk'])]:
  scale,zero=detail['quantization'];q[name]=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
 states={f'cache{i}':np.full(bi[f'cache{i}']['shape'],bi[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
 codes={name:np.empty(len(values)//3,np.uint8) for name in MODELS};decoders={name:Events() for name in MODELS};events={name:[] for name in MODELS};emitted=[];suppressed=[];last=-1000;started=time.perf_counter()
 for call,start in enumerate(range(0,len(values),3)):
  step=call+1;a.set_tensor(ai['index'],q['c161'][None,start:start+3]);a.invoke();codes['c161'][call]=a.get_tensor(ao['index'])[0,0]
  result=runner(chunk=q['binary40'][None,start:start+3,None,:],**states);codes['binary40'][call]=int(result['probability'][0,0,0,0])+128
  for i in range(7):
   s,z=bo[f'cache{i}_out']['quantization'];ts,tz=bi[f'cache{i}']['quantization'];v=result[f'cache{i}_out']
   states[f'cache{i}']=v.copy() if (s,z)==(ts,tz) else np.clip(np.rint((v.astype(np.float64)-z)*s/ts)+tz,-128,127).astype(np.int8)
  for name in ['c161','binary40']:
   if decoders[name].update(step,codes[name][call]):
    event=dict(step=step,relative_sample=int(clock[call]));events[name].append(event);candidate=dict(**event,model=name)
    if step-last>=67:emitted.append(candidate);last=step
    else:suppressed.append(candidate)
 return codes,events,emitted,suppressed,dict(interleaved_inference_and_decoder_seconds=time.perf_counter()-started,calls=len(values)//3,quantized_input_sha256={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in q.items()})
def main():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();plan=read(LOCK)
 for p,h in plan['files'].items():assert sha(p)==h,p
 for name,version in plan['versions'].items():assert importlib.metadata.version(name)==version
 assert shutil.disk_usage('.').free>100*1024**3;B.mkdir(exist_ok=False);state=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],status='running',plan_sha256=sha(LOCK),completed_rows=0)
 def save():(B/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
  reference=read(R/'4090-c161-binary40-parallel-events-original-v2.json');reports={};started=time.monotonic();save()
  for c in COHORTS:
   folder=B/c;folder.mkdir();records=[]
   for item,ref in zip(inputs(c),reference['cohorts'][c]['records'],strict=True):
    index,source,pcm,oldpath,oldhash,q,original,timeline=item
    assert index==ref['index']
    if c!='continuous':assert source==ref['source']
    combined=np.concatenate((np.zeros(81600,np.int16),pcm));raw=features(combined);values,support,_=prepare(raw,len(combined));clock=np.minimum(((support+479)//480)*480,len(combined))-81600
    if original:
     assert hashlib.sha256(raw.tobytes()).hexdigest()==original['raw_frontend_sha256'] and hashlib.sha256(values.tobytes()).hexdigest()==original['input_sha256']
    codes,events,emitted,suppressed,timing=dual(values,clock)
    assert timing['quantized_input_sha256']['binary40']==q['quantized_input_sha256']
    for name,spec in ref['paths'].items():
     assert sha(spec['path'])==spec['sha256']
     with np.load(spec['path'],allow_pickle=False) as z:assert np.array_equal(codes[name],z['uint8']) and np.array_equal(clock,z[spec['clock']]-spec['offset'])
     assert events[name]==ref['saved'][name]
    assert emitted==ref['emitted'] and suppressed==ref['suppressed']
    trace=folder/f'{ref["phase"]}-{index:06d}.npz';np.savez_compressed(trace,c161_uint8=codes['c161'],binary40_uint8=codes['binary40'],relative_block_return=clock)
    row=dict(index=index,phase=ref['phase'],source=ref['source'],trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),events=events,emitted=emitted,suppressed=suppressed,timing=timing,per_model_codes_exact=True,online_events_exact=True)
    records.append(row)
    with (folder/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    state.update(cohort=c,index=index,completed_rows=state['completed_rows']+1);save();print(json.dumps(dict(cohort=c,index=index,rows=state['completed_rows'],exact=True)),flush=True)
   report=folder/'result.json';report.write_text(json.dumps(dict(cohort=c,records=records,all_passed=True),indent=2)+'\n');reports[c]=dict(path=str(report),sha256=sha(report),rows=len(records))
  assert state['completed_rows']==1760;out=dict(schema='c161-binary40-parallel-runtime-original-v1',all_passed=True,cohorts=reports,plan_sha256=sha(LOCK),elapsed_seconds=time.monotonic()-started,total_rows=1760,models=MODELS,optimizer_updates=0,limits='Host BUILTIN_REF1thread interleaved dual inference; all per-model codes and online events reproduce saved references. Fresh interpreters only at complete stream start; no prime-boundary reset. Host timings include Python/decoder, not ESP32-P4 arena/realtime/first-command/stability evidence.')
  with (R/'4090-c161-binary40-parallel-runtime-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
  state['status']='completed-await-independent-review'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(B/'failure.log').write_text(state['error']);raise
 finally:save()
if __name__=='__main__':main()
