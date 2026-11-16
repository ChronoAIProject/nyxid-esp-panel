import json,sys,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
sys.path[:0]=['research/wakeword_p4/training','research/wakeword_p4/experiments'];from features import features;from uint8_stream_detector_v1 import Uint8StreamDetector;from uint8_event_policy_v1 import event_steps
base=Path('.local/wakeword/backgrounds');m=json.load(open(base/'manifest.json'));rows=[r for r in m['records'] if True];d=Uint8StreamDetector('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite');out=[]
for r in rows:
 p=base/r['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];w,s=sf.read(p,dtype='int16');a=d.predict(features(w));ev=event_steps(a.tolist(),.91796875);out.append({'path':r['path'],'seconds':r['seconds'],'frames':len(a),'events':len(ev),'sha256':r['sha256']})
rep={'schema':'kokoro-v11-v30-background-replay-v2','model_sha256':'04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c','threshold':.91796875,'smoothing_steps':5,'records':out,'seconds':sum(x['seconds'] for x in out),'events':sum(x['events'] for x in out),'scope':'Existing validation background; no positive/acceptance access; synthetic-only FAR proxy'};Path('research/wakeword_p4/results/4090-kokoro-v11-v30-background-replay-v2.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({'records':len(out),'seconds':rep['seconds'],'events':rep['events']}))
