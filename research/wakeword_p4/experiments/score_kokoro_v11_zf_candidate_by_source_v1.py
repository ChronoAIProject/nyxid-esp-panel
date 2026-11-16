import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
import sys;sys.path.insert(0,'research/wakeword_p4/training');sys.path.insert(0,'research/wakeword_p4/experiments')
from features import features
from uint8_stream_detector_v1 import Uint8StreamDetector
from uint8_event_policy_v1 import event_steps
model='.local/wakeword/kokoro-v11-candidate-zf-teacher60-v1/hey_nyx.tflite'; d=Uint8StreamDetector(model)
out=[]
for root,source in [('.local/wakeword/kokoro-v11-zf-development-v1','zf_training_diagnostic'),('.local/wakeword/kokoro-v11-development-boundary-v2','sol_independent_validation')]:
 for r in json.loads((Path(root)/'records.json').read_text()):
  if r['phrase_id'] not in ('nyx','official'): continue
  if source=='sol_independent_validation' and r['voice']!='af_sol': continue
  w,s=sf.read(Path(root)/r['pcm_path'],dtype='int16'); a=features(w); d.reset(); raw=d.predict(a); ev=event_steps(raw.tolist(),.95)
  out.append({'source':source,'request_id':r['request_id'],'frames':len(a),'max_uint8':int(raw.max()),'events':ev,'event':bool(ev)})
rep={'schema':'kokoro-v11-zf-candidate-source-replay-v1','model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),'threshold':.95,'smoothing_steps':5,'records':out,'summary':{s:{'records':sum(x['source']==s for x in out),'events':sum(x['source']==s and x['event'] for x in out),'max':max(x['max_uint8'] for x in out if x['source']==s)} for s in sorted({x['source'] for x in out})},'scope':'zf rows are training diagnostic; Sol rows are independent development validation; no acceptance or promotion claim'}
Path('research/wakeword_p4/results/4090-kokoro-v11-zf-candidate-source-replay-v1.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep['summary']))
