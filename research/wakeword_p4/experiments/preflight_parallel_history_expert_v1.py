"""Fixed third-expert saved-score feasibility; not actual triple inference."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from run_parallel_context_streams_v1 import material
from collect_multispeaker_formal_v1 import prepare
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-history-expert-preflight-plan-v1.json';OUT=Path('.local/wakeword/parallel-history-expert-preflight-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def merge(own,priority):
 candidates=sorted([dict(e,model=n) for n,es in own.items() for e in es],key=lambda e:(e['step'],priority.index(e['model'])));emitted=[];suppressed=[]
 for e in candidates:
  if not emitted or e['step']-emitted[-1]['step']>=67:emitted.append(e)
  else:suppressed.append(e)
 return emitted,suppressed
def lock():
 p=read(R/'4090-parallel-context-streams-plan-v1.json');q=read(R/'4090-parallel-context-streams-original-v1.json');assert read(R/'4090-parallel-context-streams-review-v1.json')['all_passed']
 export=R/'4090-frozen-history-teacher-teacher-export-original-v1.json';model=read(export)['model'];assert model['bytes']==64800 and model['sha256']=='d92b95278334043ff70c20c6e8f267e13fe83acac7dbcf705cc2177df9130950' and sha(model['path'])==model['sha256']
 files=dict(p['files'])
 for path in [Path(__file__),Path(__file__).with_name('parallel_history_expert_preflight_v1.md'),Path(__file__).with_name('run_parallel_context_streams_v1.py'),Path(__file__).with_name('binary40_stream_predict_v1.py'),R/'4090-parallel-context-streams-plan-v1.json',R/'4090-parallel-context-streams-original-v1.json',R/'4090-parallel-context-streams-review-v1.json',export,Path(model['path'])]+[Path(r['trace']['path']) for r in q['streams']]:files[str(path)]=sha(path)
 out=dict(schema='parallel-history-expert-preflight-plan-v1',files=files,model=model,priority=['c161','binary40','history_expert'],policy='Read fixed proposal; allthree complete context streams, no sources/parameters/parents sweep. Infer only exact frozen history expert and authenticate original two raw streams, then saved-score arbitration. Not actual interleaved triple execution or P4 resources. No optimizer/synthesis/acceptance/new background exposure.')
 with PLAN.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from binary40_stream_predict_v1 import predict
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
 source=read(R/'4090-parallel-context-streams-plan-v1.json');old=read(R/'4090-parallel-context-streams-original-v1.json');OUT.mkdir(exist_ok=False);rows=[]
 for stream,reference in zip(source['streams'],old['streams'],strict=True):
  assert stream['name']==reference['name'];pcm,timeline=material(stream,8000);whole=np.r_[np.zeros(81600,np.int16),pcm];raw=features(whole);values,support,_=prepare(raw,len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600
  assert hashlib.sha256(pcm.tobytes()).hexdigest()==reference['pcm_sha256'] and timeline==reference['timeline']
  assert hashlib.sha256(raw.tobytes()).hexdigest()==reference['raw_frontend_sha256'] and hashlib.sha256(values.tobytes()).hexdigest()==reference['prepared_features_sha256']
  _,codes,quantization=predict(values,p['model']['path'],runtime);own={};arrays={}
  with np.load(reference['trace']['path'],allow_pickle=False) as z:
   assert np.array_equal(clock,z['relative_block_return'])
   for name in ['c161','binary40']:
    arrays[name+'_uint8']=z[name+'_uint8'].copy();own[name]=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(arrays[name+'_uint8'],.91796875)];assert own[name]==reference['events'][name]
  assert merge(own,p['priority'][:2])==(reference['emitted'],reference['suppressed'])
  arrays['history_expert_uint8']=codes;arrays['relative_block_return']=clock;own['history_expert']=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)];emitted,suppressed=merge(own,p['priority']);trace=OUT/(stream['name']+'.npz');np.savez_compressed(trace,**arrays)
  row=dict(name=stream['name'],timeline=timeline,events=own,emitted=emitted,suppressed=suppressed,existing_dual_emitted=reference['emitted'],quantization=quantization,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)));rows.append(row);print(json.dumps(dict(name=stream['name'],own_events={n:len(es) for n,es in own.items()},triple_emitted=len(emitted))),flush=True)
 out=dict(schema='parallel-history-expert-preflight-original-v1',plan_sha256=sha(PLAN),streams=rows,optimizer_updates=0,policy=p['policy'])
 with (R/'4090-parallel-history-expert-preflight-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
