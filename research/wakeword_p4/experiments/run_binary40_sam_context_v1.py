"""Final arm on all three complete physical context streams, fixed protocol."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from run_parallel_context_streams_v1 import material
from collect_multispeaker_formal_v1 import prepare
from binary40_stream_predict_v1 import predict
from uint8_event_policy_v1 import event_steps
R=Path('research/wakeword_p4/results');ARM=sys.argv[2];assert ARM in ['sam'];PLAN=R/f'4090-binary40-sam-{ARM}-context-plan-v1.json';OUT=Path(f'.local/wakeword/binary40-sam-{ARM}-context-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def plan():
 source=R/'4090-parallel-context-streams-plan-v1.json';reference=R/'4090-parallel-context-streams-original-v1.json';export=R/f'4090-binary40-sam-{ARM}-export-original-v1.json';e=read(export);files=dict(read(source)['files'])
 for p in [Path(__file__),source,reference,export,Path(e['model']['path']),Path(__file__).with_name('binary40_stream_predict_v1.py'),Path(__file__).with_name('uint8_event_policy_v1.py')]:files[str(p)]=sha(p)
 with PLAN.open('x') as f:json.dump(dict(arm=ARM,files=files,model=e['model'],policy='All3whole48clipcontextstreams, original source/order/8000gap/one81600zeroPCMprime/no boundaryreset. FinalonearmINT8, fixed235/256/5/67/hHalfrearm, ownstatequantization. Keepcomplete uint8/nativeclock/events/container identities; compareoriginalC161/B40references without reinference. Consumedsynthetic, no FAR/new exposure/firstcommand/P4claim.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 OUT.mkdir(exist_ok=False);rows=[]
 for stream,ref in zip(read(R/'4090-parallel-context-streams-plan-v1.json')['streams'],read(R/'4090-parallel-context-streams-original-v1.json')['streams'],strict=True):
  assert stream['name']==ref['name'];pcm,timeline=material(stream,8000);whole=np.r_[np.zeros(81600,np.int16),pcm];raw=features(whole);values,support,_=prepare(raw,len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600;assert timeline==ref['timeline'] and hashlib.sha256(raw.tobytes()).hexdigest()==ref['raw_frontend_sha256'] and hashlib.sha256(values.tobytes()).hexdigest()==ref['prepared_features_sha256']
  _,codes,quantization=predict(values,p['model']['path'],runtime);events=[dict(step=int(s),relative_sample=int(clock[s-1])) for s in event_steps(codes,.91796875)];groups={};outside=[]
  for t in timeline:
   if t['kind']=='clip':
    es=[e for e in events if t['start']<e['relative_sample']<=t['end']];base={n:[e for e in ref['events'][n] if t['start']<e['relative_sample']<=t['end']] for n in ['c161','binary40']};g=groups.setdefault(t['group'],dict(clips=0,hits=0,c161_hits=0,original_binary40_hits=0));g['clips']+=1;g['hits']+=bool(es);g['c161_hits']+=bool(base['c161']);g['original_binary40_hits']+=bool(base['binary40'])
  for e in events:
   owner=next((t for t in timeline if t['start']<e['relative_sample']<=t['end']),None)
   if owner is None or owner['kind']!='clip':outside.append(dict(event=e,owner=owner))
  trace=OUT/(stream['name']+'.npz');np.savez_compressed(trace,uint8=codes,relative_block_return=clock);row=dict(name=stream['name'],timeline=timeline,groups=groups,events=events,outside_clip_events=outside,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)),quantization=quantization);rows.append(row);print(json.dumps(dict(name=row['name'],groups=groups)),flush=True)
 out=dict(schema='binary40-sam-context-original-v1',arm=ARM,plan_sha256=sha(PLAN),all_passed=True,streams=rows,limits=p['policy'])
 with (R/f'4090-binary40-sam-{ARM}-context-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
