"""Three complete physical context streams; existing short-gap protocol, no resets."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from run_c161_binary40_parallel_v1 import dual
from collect_multispeaker_formal_v1 import prepare
R=Path('research/wakeword_p4/results');PLAN=R/'4090-parallel-context-streams-plan-v1.json';OUT=Path('.local/wakeword/parallel-context-streams-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lock():
 paths=[Path(__file__)]+[Path('research/wakeword_p4/experiments')/f for f in ['run_c161_binary40_parallel_v1.py','c161_binary40_parallel_sources_v1.py','replay_c161_binary40_parallel_events_v2.py','collect_multispeaker_formal_v1.py','uint8_event_policy_v1.py']]+[Path('research/wakeword_p4/training/features.py')]
 groups=[]
 for prefix,parts in [('parallel-prospective',[('builtin',0)]),('parallel-melo-context',[('melo-US',0),('melo-BR',48)])]:
  pp=R/f'4090-{prefix}-replay-plan-v1.json';rp=R/f'4090-{prefix}-replay-original-v1.json';vp=R/f'4090-{prefix}-review-original-v1.json';paths.extend([pp,rp,vp]);p=read(pp);result=read(rp);assert read(vp)['all_passed']
  for name,offset in parts:
   items=[]
   # Fixed cycle: wake command, two ordinary sentences, one name context.
   order=[k for i in range(12) for k in [i,12+2*i,13+2*i,36+i]];assert sorted(order)==list(range(48))
   for k in order:
    gen=p['rows'][offset+k];old=result['records'][offset+k];assert gen['index']==old['index']==offset+k
    wav,=[x['path'] for x in gen['files'] if x['path'].endswith('.wav')];paths.append(Path(wav));assert sha(wav)==next(x['sha256'] for x in gen['files'] if x['path']==wav)
    items.append(dict(text_index=k,source_index=offset+k,request=gen['request'],wav=wav,pcm_sha256=old['pcm_sha256'],isolated_events=old['events'],isolated_emitted=old['emitted']))
   groups.append(dict(name=name,items=items))
 models=read(R/'4090-parallel-prospective-language-plan-v1.json')['models']
 for m in models:assert sha(m['path'])==m['sha256'];paths.append(Path(m['path']))
 p=dict(schema='parallel-context-streams-plan-v1',files={str(p):sha(p) for p in paths},versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},streams=groups,gap_samples=8000,prime_samples=81600,profile=dict(threshold=.91796875,smoothing=5,refractory=67,native_samples=480),policy='Three whole48clip physical streams, original shortgap8000between clips only, fixed interleave target/twoordinary/name. One continuous81600zeroPCM prime, no model/frontend/decoder/arbiter reset at any clip or prime boundary. Original C161/B40, online independent detectors/shared67call policy. Complete waveforms and raw scores/clocks/events retained, exact (start,end] block-return container ownership. Gap events separate, not automatically false or target; container counts not phonetic onset/first-command proof. Reused consumed audio, no new independent speaker/acceptance/background exposure, synthesis, training, threshold/order/gap sweep or tuning.')
 with PLAN.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def material(stream,gap):
 import soundfile as sf
 parts=[];timeline=[];position=0
 for i,item in enumerate(stream['items']):
  pcm,rate=sf.read(item['wav'],dtype='int16');assert rate==16000 and pcm.ndim==1 and hashlib.sha256(pcm.tobytes()).hexdigest()==item['pcm_sha256']
  end=position+len(pcm);timeline.append(dict(kind='clip',start=position,end=end,text_index=item['text_index'],group=item['request']['group'],text=item['request']['text']));parts.append(pcm);position=end
  if i+1<len(stream['items']):timeline.append(dict(kind='gap',start=position,end=position+gap,after_text_index=item['text_index']));parts.append(np.zeros(gap,np.int16));position+=gap
 return np.concatenate(parts),timeline
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=read(PLAN)
 for path,h in p['files'].items():assert sha(path)==h,path
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 assert shutil.disk_usage('.').free>100*1024**3
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 OUT.mkdir(exist_ok=False);rows=[]
 for stream in p['streams']:
  pcm,timeline=material(stream,p['gap_samples']);whole=np.concatenate([np.zeros(81600,np.int16),pcm]);raw=features(whole);values,support,_=prepare(raw,len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600
  codes,events,emit,suppress,timing=dual(values,clock);trace=OUT/(stream['name']+'.npz');np.savez_compressed(trace,c161_uint8=codes['c161'],binary40_uint8=codes['binary40'],relative_block_return=clock)
  row=dict(name=stream['name'],pcm_samples=len(pcm),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),timeline=timeline,events=events,emitted=emit,suppressed=suppress,timing=timing,raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),prepared_features_sha256=hashlib.sha256(values.tobytes()).hexdigest(),trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)))
  rows.append(row);(OUT/(stream['name']+'.json')).write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(dict(name=row['name'],events={k:len(v) for k,v in events.items()},emitted=len(emit))),flush=True)
 out=dict(schema='parallel-context-streams-original-v1',plan_sha256=sha(PLAN),streams=rows,limits=p['policy'])
 with (R/'4090-parallel-context-streams-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
