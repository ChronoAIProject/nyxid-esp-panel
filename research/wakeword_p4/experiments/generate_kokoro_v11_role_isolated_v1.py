import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from kokoro import KModel, KPipeline

SR=24000
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--role',default='development'); ap.add_argument('--lock',type=Path,required=True); a=ap.parse_args()
  plan=json.loads(a.plan.read_text()); req=[r for r in plan['requests'] if r['role']==a.role]
  if a.output.exists() and any(a.output.iterdir()): raise SystemExit('output must be fresh')
  a.output.mkdir(parents=True,exist_ok=True)
  device='cuda' if torch.cuda.is_available() else 'cpu'; model=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to(device).eval()
  pipes={'af':KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M-v1.1-zh',model=model),'bf':KPipeline(lang_code='b',repo_id='hexgrad/Kokoro-82M-v1.1-zh',model=model)}
  records=[]; start=time.time()
  for i,r in enumerate(req):
    g=pipes[r['voice'][:2]](r['text'],voice=r['voice'],speed=r['speed']); out=next(g); native=np.asarray(out.audio,dtype=np.float32)
    if native.ndim!=1 or not np.isfinite(native).all() or native.size==0: raise RuntimeError(f'quality failure {r["request_id"]}')
    stem=f'{i:04d}-{r["request_id"]}'; np.save(a.output/(stem+'.native.npy'),native,allow_pickle=False); pcm=resample_poly(native,2,3); wav=a.output/(stem+'.wav'); sf.write(wav,pcm,16000,subtype='PCM_16')
    records.append({**r,'native_path':str((stem+'.native.npy')),'pcm_path':str(stem+'.wav'),'native_sha256':sha(a.output/(stem+'.native.npy')),'pcm_sha256':sha(wav),'native_samples':int(native.size),'native_sample_rate':SR,'sample_rate':16000,'pcm_samples':int(len(pcm)),'phoneme_metadata':None,'duration_metadata':None})
    if i%8==0: print(json.dumps({'role':a.role,'done':i+1,'total':len(req),'device':device}),flush=True)
  (a.output/'records.json').write_text(json.dumps(records,indent=2)+'\n')
  print(json.dumps({'role':a.role,'records':len(records),'seconds':round(time.time()-start,1),'device':device}),flush=True)
if __name__=='__main__': main()
