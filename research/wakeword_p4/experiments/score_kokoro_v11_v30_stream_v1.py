import json,hashlib,sys
from pathlib import Path
import numpy as np,soundfile as sf
sys.path.insert(0,'research/wakeword_p4/training');sys.path.insert(0,'research/wakeword_p4/experiments')
from features import features
from uint8_stream_detector_v1 import Uint8StreamDetector
root=Path('.local/wakeword/kokoro-v11-role-isolated-development-v2'); rows=json.loads((root/'records.json').read_text()); d=Uint8StreamDetector('.local/wakeword/model-gpu-v30-ami-extra/hey_nyx.tflite');out=[]
for r in rows:
 w,s=sf.read(root/r['pcm_path'],dtype='int16'); raw=features(w); d.reset(); a=d.predict(raw); d.reset(); b=d.predict(raw); assert np.array_equal(a,b); out.append({'request_id':r['request_id'],'phrase_id':r['phrase_id'],'label':int(r['phrase_id'] in ('nyx','official')),'frames':len(raw),'max_uint8':int(a.max()),'events_ge_235':int((a>=235).sum()),'reset_repeat_exact':True})
rep={'schema':'kokoro-v11-v30-stream-replay-v1','model_sha256':'04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c','records':out,'scope':'Synthetic development only; no threshold selection or promotion'};Path('research/wakeword_p4/results/4090-kokoro-v11-v30-stream-replay-v1.json').write_text(json.dumps(rep,indent=2)+'\n');print('records',len(out),'positive',sum(x['label'] for x in out),'events',sum(x['events_ge_235'] for x in out))
