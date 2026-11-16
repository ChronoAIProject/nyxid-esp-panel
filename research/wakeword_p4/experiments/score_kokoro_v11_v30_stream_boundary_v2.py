import json,hashlib,sys
from pathlib import Path
import numpy as np,soundfile as sf
sys.path.insert(0,'research/wakeword_p4/training');sys.path.insert(0,'research/wakeword_p4/experiments')
from features import features
from uint8_stream_detector_v1 import Uint8StreamDetector
from uint8_event_policy_v1 import event_steps
root=Path('.local/wakeword/kokoro-v11-development-boundary-v2'); rows=json.loads((root/'records.json').read_text()); d=Uint8StreamDetector('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite');out=[]
for r in rows:
 w,s=sf.read(root/r['pcm_path'],dtype='int16'); raw=features(w); d.reset(); a=d.predict(raw); d.reset(); b=d.predict(raw); assert np.array_equal(a,b); ev=event_steps(a.tolist(),0.91796875); out.append({'request_id':r['request_id'],'phrase_id':r['phrase_id'],'label':int(r['phrase_id'] in ('nyx','official')),'frames':len(raw),'max_uint8':int(a.max()),'raw_frames_ge_235':int((a>=235).sum()),'smoothed_events':len(ev),'event_steps':ev,'reset_repeat_exact':True})
rep={'schema':'kokoro-v11-v30-stream-replay-boundary-v2','model_sha256':'04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c','threshold':0.91796875,'smoothing_steps':5,'records':out,'summary':{'positive_records':sum(x['label'] for x in out),'negative_records':sum(not x['label'] for x in out),'positive_event_records':sum(bool(x['smoothed_events']) and x['label'] for x in out),'negative_event_records':sum(bool(x['smoothed_events']) and not x['label'] for x in out)},'scope':'Synthetic development only; threshold is frozen v30 policy, not newly selected'};Path('research/wakeword_p4/results/4090-kokoro-v11-v30-stream-replay-boundary-v2.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep['summary']))
