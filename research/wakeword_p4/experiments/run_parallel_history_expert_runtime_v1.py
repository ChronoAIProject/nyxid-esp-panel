"""Fixed three-interpreter native-call execution; authenticate all saved references."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import hashlib,importlib.metadata,io,json,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import numpy as np
from run_c161_binary40_parallel_v1 import Events,MODELS as DUAL
from c161_binary40_parallel_sources_v1 import inputs
from run_parallel_context_streams_v1 import material
from collect_multispeaker_formal_v1 import prepare
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/parallel-history-expert-runtime-v1');PLAN=R/'4090-parallel-history-expert-runtime-plan-v1.json'
MODELS={**DUAL,'history_expert':'.local/wakeword/frozen-history-teacher-int8-v1/teacher/int8.tflite'}
PRIORITY=list(MODELS)
COHORTS=['builtin','daily','continuous','primed','jenny','joe','melo','mac']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def triple(values,clock):
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 runtimes={};q={}
 for name,path in MODELS.items():
  it=Interpreter(model_path=path,num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF);it.allocate_tensors()
  if name=='c161':
   inp=it.get_input_details()[0];out=it.get_output_details()[0];assert list(inp['shape'])==[1,3,40] and out['dtype']==np.uint8 and out['quantization']==(1/256,0)
   runtimes[name]=(it,inp,out)
  else:
   runner=it.get_signature_runner();bi=runner.get_input_details();bo=runner.get_output_details();inp=bi['chunk'];assert list(inp['shape'])==[1,3,1,40] and bo['probability']['dtype']==np.int8 and bo['probability']['quantization']==(1/256,-128)
   states={f'cache{i}':np.full(bi[f'cache{i}']['shape'],bi[f'cache{i}']['quantization'][1],np.int8) for i in range(7)}
   runtimes[name]=(it,runner,bi,bo,states)
  assert inp['dtype']==np.int8;scale,zero=inp['quantization'];q[name]=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
 codes={n:np.empty(len(values)//3,np.uint8) for n in PRIORITY};decoders={n:Events() for n in PRIORITY};events={n:[] for n in PRIORITY};emitted=[];suppressed=[];last=-1000;started=time.perf_counter()
 for call,start in enumerate(range(0,len(values),3)):
  step=call+1
  for name in PRIORITY:
   if name=='c161':
    it,inp,out=runtimes[name];it.set_tensor(inp['index'],q[name][None,start:start+3]);it.invoke();codes[name][call]=it.get_tensor(out['index'])[0,0]
   else:
    it,runner,bi,bo,states=runtimes[name];result=runner(chunk=q[name][None,start:start+3,None,:],**states);codes[name][call]=int(result['probability'][0,0,0,0])+128
    for i in range(7):
     s,z=bo[f'cache{i}_out']['quantization'];ts,tz=bi[f'cache{i}']['quantization'];v=result[f'cache{i}_out'];states[f'cache{i}']=v.copy() if (s,z)==(ts,tz) else np.clip(np.rint((v.astype(np.float64)-z)*s/ts)+tz,-128,127).astype(np.int8)
  # All three interpreters finish this native call before decoding fixed-priority events.
  for name in PRIORITY:
   if decoders[name].update(step,codes[name][call]):
    e=dict(step=step,relative_sample=int(clock[call]));events[name].append(e);candidate=dict(**e,model=name)
    if step-last>=67:emitted.append(candidate);last=step
    else:suppressed.append(candidate)
 return codes,events,emitted,suppressed,dict(calls=len(clock),interleaved_inference_and_decoder_seconds=time.perf_counter()-started,quantized_input_sha256={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in q.items()})
def lock():
 files={}
 for n in ['c161-binary40-parallel-runtime','parallel-history-expert-scope','parallel-history-expert-preflight']:
  p=R/f'4090-{n}-plan-v1.json';files.update(read(p)['files']);files[str(p)]=sha(p)
 for n in ['parallel-history-expert-scope-original','parallel-history-expert-scope-review','parallel-history-expert-preflight-original','parallel-history-expert-preflight-review','parallel-context-streams-original','parallel-context-streams-plan']:
  p=R/f'4090-{n}-v1.json';files[str(p)]=sha(p)
 for row in read(R/'4090-parallel-history-expert-preflight-original-v1.json')['streams']:files[row['trace']['path']]=row['trace']['sha256']
 for p in [Path(__file__),Path(__file__).with_name('review_parallel_history_expert_runtime_v1.py')]+list(map(Path,MODELS.values())):files[str(p)]=sha(p)
 # Authenticate closure now and again at execution.
 for p,h in files.items():assert sha(p)==h,p
 out=dict(schema='parallel-history-expert-runtime-plan-v1',files=files,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},models={n:dict(path=p,bytes=Path(p).stat().st_size,sha256=sha(p)) for n,p in MODELS.items()},priority=PRIORITY,total_rows=1763,policy='Exact fixed C161 + original Binary40 + frozen-history-teacher treatment. Actual all3 interpreters invoked every native call, then own235/256/5/67/half-rearm and fixed priority shared67call emitted cooldown. One81600zeroPCM prime per fullstream; no boundary resets. All1760original sources plus3newcontext streams, exact rawuint8/nativeclock/own/emitted/suppressed against authenticated saved references. No optimizer, parent/policy/threshold/seed sweep or fourth expert. Fresh interpreters at fullstream starts only. Host BUILTIN_REF1thread, CPU numerical2. Not P4 resources/realtime/first-command/acceptance evidence. Archive every raw trace and record; independent decoder/event/member audit required.')
 with PLAN.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 assert shutil.disk_usage('.').free>100*1024**3;OUT.mkdir(exist_ok=False)
 state=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],status='running',completed_rows=0,plan_sha256=sha(PLAN));started=time.monotonic()
 def save():(OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  save();sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
  base=read(R/'4090-c161-binary40-parallel-events-original-v2.json');scope=read(R/'4090-parallel-history-expert-scope-original-v1.json');reports={}
  def execute(pcm,ref,folder,label,expected_arrays,original=None,quant=None):
   whole=np.r_[np.zeros(81600,np.int16),pcm];raw=features(whole);values,support,_=prepare(raw,len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600
   if original:assert hashlib.sha256(raw.tobytes()).hexdigest()==original['raw_frontend_sha256'] and hashlib.sha256(values.tobytes()).hexdigest()==original['input_sha256']
   codes,events,emitted,suppressed,timing=triple(values,clock)
   if quant:assert timing['quantized_input_sha256']['binary40']==quant['quantized_input_sha256']
   for name,(old,oldclock) in expected_arrays.items():assert np.array_equal(codes[name],old) and np.array_equal(clock,oldclock),(label,name)
   assert events==ref.get('saved',ref.get('events')) and emitted==ref['emitted'] and suppressed==ref['suppressed'],label
   trace=folder/(label+'.npz');np.savez_compressed(trace,**{n+'_uint8':v for n,v in codes.items()},relative_block_return=clock)
   row=dict(label=label,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),events=events,emitted=emitted,suppressed=suppressed,timing=timing,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),prepared_features_sha256=hashlib.sha256(values.tobytes()).hexdigest(),exact=True)
   with (folder/'records.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   state['completed_rows']+=1;state['last']=str(trace);save();print(json.dumps(dict(rows=state['completed_rows'],trace=str(trace),exact=True)),flush=True)
   return row
  for c in COHORTS:
   folder=OUT/c;folder.mkdir();rows=[]
   archive=scope['cohorts'][c]['records'][0]['teacher_trace']['archive']
   with tarfile.open(archive,'r:gz') as tf:
    for item,ref,old in zip(inputs(c),scope['cohorts'][c]['records'],base['cohorts'][c]['records'],strict=True):
     index,source,pcm,oldpath,oldhash,q,original,timeline=item;assert index==ref['index']==old['index'];assert source==ref['source'] if c!='continuous' else timeline==ref['timeline']
     expected={}
     for n,sp in old['paths'].items():
      assert sha(sp['path'])==sp['sha256']
      with np.load(sp['path'],allow_pickle=False) as z:expected[n]=(z['uint8'].copy(),z[sp['clock']]-sp['offset'])
     spec=ref['teacher_trace'];data=tf.extractfile(spec['member']).read();assert hashlib.sha256(data).hexdigest()==spec['sha256']
     with np.load(io.BytesIO(data),allow_pickle=False) as z:
      ck='block_return_sample' if c=='melo' else 'real_block_return_sample' if c in ['primed','jenny','joe'] else 'relative_block_return';expected['history_expert']=(z['uint8'].copy(),z[ck]-(81600 if c=='melo' else 0))
     row=execute(pcm,ref,folder,f'{ref["phase"]}-{index:06d}',expected,original,q);row.update(index=index,phase=ref['phase'],source=ref['source']);rows.append(row)
   rp=folder/'result.json';rp.write_text(json.dumps(dict(cohort=c,records=rows,all_passed=True),indent=2)+'\n');reports[c]=dict(path=str(rp),sha256=sha(rp),rows=len(rows))
  folder=OUT/'context';folder.mkdir();rows=[];context=read(R/'4090-parallel-context-streams-plan-v1.json');oldcontext=read(R/'4090-parallel-context-streams-original-v1.json')
  for stream,ref,original in zip(context['streams'],read(R/'4090-parallel-history-expert-preflight-original-v1.json')['streams'],oldcontext['streams'],strict=True):
   assert stream['name']==ref['name']==original['name'];pcm,timeline=material(stream,8000);assert timeline==ref['timeline'] and hashlib.sha256(pcm.tobytes()).hexdigest()==original['pcm_sha256'];expected={}
   with np.load(ref['trace']['path'],allow_pickle=False) as z:
    for n in PRIORITY:expected[n]=(z[n+'_uint8'].copy(),z['relative_block_return'].copy())
   o=dict(raw_frontend_sha256=original['raw_frontend_sha256'],input_sha256=original['prepared_features_sha256']);row=execute(pcm,ref,folder,stream['name'],expected,o);row.update(name=stream['name'],timeline=timeline);rows.append(row)
  rp=folder/'result.json';rp.write_text(json.dumps(dict(records=rows,all_passed=True),indent=2)+'\n');reports['context']=dict(path=str(rp),sha256=sha(rp),rows=len(rows))
  assert state['completed_rows']==1763
  out=dict(schema='parallel-history-expert-runtime-original-v1',plan_sha256=sha(PLAN),cohorts=reports,total_rows=1763,elapsed_seconds=time.monotonic()-started,all_passed=True,optimizer_updates=0,policy=p['policy'])
  with (R/'4090-parallel-history-expert-runtime-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
  state['status']='completed-await-independent-review'
 except Exception:
  state['status']='failed-preserved';state['error']=traceback.format_exc();(OUT/'failure.log').write_text(state['error']);raise
 finally:save()
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
