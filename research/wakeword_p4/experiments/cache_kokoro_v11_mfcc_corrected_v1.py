import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
import sys;sys.path.insert(0,'research/wakeword_p4/training');from features import features
src=Path('.local/wakeword/kokoro-v11-role-isolated-development-v2-corrected');out=Path('.local/wakeword/kokoro-v11-mfcc-corrected-v1');out.mkdir(exist_ok=False);rows=json.loads((src/'records.json').read_text());rec=[]
for i,r in enumerate(rows):
 w,s=sf.read(src/r['pcm_path'],dtype='int16');z=features(w);p=out/f'{i:04d}.npy';np.save(p,z,allow_pickle=False);rec.append({'request_id':r['request_id'],'label':int(r['phrase_id'] in ('nyx','official')),'phrase_id':r['phrase_id'],'carrier':r['carrier'],'text':r['text'],'pcm_sha256':r['pcm_sha256'],'feature_path':str(p),'feature_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':len(z),'width':40,'sample_rate':16000})
(out/'manifest.json').write_text(json.dumps({'schema':'kokoro-v11-mfcc-corrected-v1','records':rec,'acceptance_excluded':True,'threshold_selection_excluded':True},indent=2)+'\n');print('records',len(rec),'frames',sum(x['frames'] for x in rec))
