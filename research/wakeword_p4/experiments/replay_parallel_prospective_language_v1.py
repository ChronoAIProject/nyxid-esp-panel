"""Complete48 fixed prospective contexts, exact original dual execution."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import hashlib,importlib.metadata,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from run_c161_binary40_parallel_v1 import dual
from collect_multispeaker_formal_v1 import prepare
R=Path('research/wakeword_p4/results');DATA=Path('.local/wakeword/parallel-prospective-language-v1');OUT=Path('.local/wakeword/parallel-prospective-replay-v1');PLAN=R/'4090-parallel-prospective-replay-plan-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lock():
 assert json.loads((DATA/'completed.json').read_text())['requests']==48
 paths=[Path(__file__),DATA/'completed.json',R/'4090-parallel-prospective-language-plan-v1.json',R/'4090-parallel-prospective-generation-lock-v1.json']
 paths.extend(Path('research/wakeword_p4/experiments')/f for f in ['run_c161_binary40_parallel_v1.py','c161_binary40_parallel_sources_v1.py','replay_c161_binary40_parallel_events_v2.py','collect_multispeaker_formal_v1.py','uint8_event_policy_v1.py'])
 paths.append(Path('research/wakeword_p4/training/features.py'))
 source=json.loads((R/'4090-parallel-prospective-language-plan-v1.json').read_text());rows=[]
 for i,request in enumerate(source['requests']):
  p=DATA/f'{i:03d}.json';row=json.loads(p.read_text());assert row['index']==i and row['request']==request;paths.append(p)
  for f in row['files']:assert sha(f['path'])==f['sha256'];paths.append(Path(f['path']))
  rows.append(row)
 for model in source['models']:assert sha(model['path'])==model['sha256'];paths.append(Path(model['path']))
 record=dict(schema='parallel-prospective-replay-plan-v1',files={str(p):sha(p) for p in paths},rows=rows,versions={n:importlib.metadata.version(n) for n in ['numpy','soundfile','ai-edge-litert','pymicro-features']},policy=source['policy'],profile=source['operating_profile'],scope='All48 once. Actual interleaved original C161/B40 and fixed online policy; no fitting, thresholds, partial subset selection or score-based synthesis retries. Save every raw dual trace/native clock/event. Intended synthetic text not phonetic truth; same consumed voice and targeted new contexts, no independent voice/background/FAR claim.')
 with PLAN.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 assert shutil.disk_usage('.').free>100*1024**3
 import soundfile as sf
 sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
 OUT.mkdir(exist_ok=False);records=[]
 for row in p['rows']:
  wav,=[x['path'] for x in row['files'] if x['path'].endswith('.wav')];pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1
  whole=np.concatenate([np.zeros(81600,np.int16),pcm]);raw=features(whole);values,support,_=prepare(raw,len(whole));clock=np.minimum(((support+479)//480)*480,len(whole))-81600
  codes,events,emitted,suppressed,timing=dual(values,clock);trace=OUT/f'{row["index"]:03d}.npz';np.savez_compressed(trace,c161_uint8=codes['c161'],binary40_uint8=codes['binary40'],relative_block_return=clock)
  record=dict(index=row['index'],request=row['request'],pcm_samples=len(pcm),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),prepared_features_sha256=hashlib.sha256(values.tobytes()).hexdigest(),events=events,emitted=emitted,suppressed=suppressed,timing=timing,trace=dict(path=str(trace),bytes=trace.stat().st_size,sha256=sha(trace)))
  records.append(record)
  with (OUT/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(dict(index=row['index'],group=row['request']['group'],events={k:len(v) for k,v in events.items()},emitted=len(emitted))),flush=True)
 out=dict(schema='parallel-prospective-replay-original-v1',completed_rows=len(records),plan_sha256=sha(PLAN),records=records,limits=p['scope'])
 with (R/'4090-parallel-prospective-replay-original-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
if __name__=='__main__':{'lock':lock,'run':run}[sys.argv[1]]()
