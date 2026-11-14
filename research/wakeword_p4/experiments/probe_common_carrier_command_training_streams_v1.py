"""Complete existing Kokoro command-negative training cohort, two raw Nyx finals, no fitting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from collect_common_carrier_raw_development_v1 import prepare, predict

R=Path('research/wakeword_p4/results')
LOCK=R/'4090-common-carrier-command-training-streams-execution-v1.json'
OUT=Path('.local/wakeword/common-carrier-command-training-streams-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(action):
 if action=='lock':
  metadata=Path('.local/wakeword/hey-common-carrier-features-v1/features.json')
  rows=json.loads(metadata.read_text());selected=[{'original_index':i,'row':r} for i,r in enumerate(rows) if r['engine']=='kokoro' and r['label']==0]
  assert len(selected)==4860 and all(e['row']['split']=='training' for e in selected)
  export=R/'4090-hey-common-carrier-raw-fp32-export-result-v1.json'
  models=[m for m in json.loads(export.read_text())['models'] if m['target']=='hey_nyx'];assert len(models)==2
  files={str(p):sha(p) for p in [metadata,export,Path(__file__),Path(__file__).with_name('collect_common_carrier_raw_development_v1.py'),Path('research/wakeword_p4/training/features.py')]}
  for e in selected:
   r=e['row'];assert sha(r['path'])==r['sha256'];files[r['path']]=r['sha256']
  for m in models:
   a=m['artifacts']['fp32'];assert sha(a['path'])==a['sha256'];files[a['path']]=a['sha256']
  with LOCK.open('x') as f:json.dump(dict(files=files,records=selected,models=models,policy='All4860 existing Kokoro training rows, original order, full PCM/native frontend, twofresh per model/row. No training, mining, selection, freshdev or acceptance. Compare to existing consumed-source traces; descriptive only.'),f,indent=2)
  print('Locked',len(selected),'rows')
 else:
  assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
  p=json.loads(LOCK.read_text())
  for f,h in p['files'].items():assert sha(f)==h,f
  OUT.mkdir(exist_ok=False)
  import soundfile as sf
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
  from features import features
  def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
  with (OUT/'records.jsonl').open('x') as log:
   for i,e in enumerate(p['records']):
    r=e['row'];pcm,rate=sf.read(r['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples']
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
    raw=features(pcm);values,support,clocks=prepare(raw,r['samples'])
    for m in p['models']:
     name=m['condition'];model=m['artifacts']['fp32']['path'];a=predict(values,model,runtime);b=predict(values,model,runtime)
     dest=OUT/f'{i:05}-{name}.npz';np.savez_compressed(dest,probability=a,probability_repeat=b,sample_end=clocks,support_unclamped=support)
     exact=a.tobytes()==b.tobytes();receipt=dict(index=i,original_index=e['original_index'],row=r,condition=name,trace=str(dest),trace_sha256=sha(dest),steps=len(a),repeat_exact=exact,frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
     log.write(json.dumps(receipt)+'\n');log.flush();assert exact
    if i%50==0:print(json.dumps(dict(completed=i+1,total=len(p['records']))),flush=True)
  (OUT/'terminal.json').write_text(json.dumps(dict(records=len(p['records'])*len(p['models']),execution_sha256=sha(LOCK),threshold_selected=False))+'\n')
if __name__=='__main__':main(sys.argv[1])
