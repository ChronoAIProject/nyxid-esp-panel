import json,hashlib
from pathlib import Path
import numpy as np,soundfile as sf
from scipy.signal import fftconvolve
src=Path('.local/wakeword/kokoro-v11-role-isolated-development-v2');out=Path('.local/wakeword/kokoro-v11-muse-augmentation-v1');out.mkdir(exist_ok=False)
rows=[r for r in json.loads((src/'records.json').read_text()) if r['phrase_id']=='muse']
rec=[]
for i,r in enumerate(rows):
 x,s=sf.read(src/r['pcm_path'],dtype='float32'); rng=np.random.default_rng(910000+i); ir=np.zeros(1200,np.float32);ir[0]=1; taps=rng.integers(20,1200,20);ir[taps]+=rng.normal(0,.12,20)*np.exp(-taps/500); y=fftconvolve(x,ir)[:len(x)]; noise=rng.normal(0,1,len(y)).astype('float32'); noise/=max(1e-8,float(np.sqrt(np.mean(noise**2)))); y=np.clip(.8*y+.02*noise,-1,1); p=out/f'{i:04d}.wav';sf.write(p,y,16000,subtype='PCM_16');rec.append({'source_request_id':r['request_id'],'phrase_id':'muse','label':0,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sample_rate':16000,'samples':len(y),'augmentation':'fixed-rir-plus-gaussian-noise','seed':910000+i,'training_only':True})
(out/'manifest.json').write_text(json.dumps({'schema':'kokoro-v11-muse-augmentation-v1','records':rec,'threshold_eligible':False,'acceptance_excluded':True},indent=2)+'\n');print('generated',len(rec))
