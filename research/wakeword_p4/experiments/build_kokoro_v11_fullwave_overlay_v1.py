import json,hashlib,shutil,subprocess
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import fftconvolve
import sys;sys.path.insert(0,'research/wakeword_p4/training');from features import features
base=Path('.local/wakeword/features-gpu-v2'); out=Path('.local/wakeword/features-kokoro-v11-fullwave-v1'); src=Path('.local/wakeword/kokoro-v11-development-boundary-v2')
if out.exists(): raise SystemExit('exists')
subprocess.run(['cp','-a','--reflink=auto',str(base),str(out)],check=True)
rows=json.loads((out/'features.json').read_text()); offset=sum(r['frames'] for r in rows); added=[]; chunks=[]
for i,r in enumerate(json.loads((src/'records.json').read_text())):
 pcm,_=sf.read(src/r['pcm_path'],dtype='float32'); positive=r['phrase_id'] in ('nyx','official')
 for variant in range(2):
  rng=np.random.default_rng(20261110+i*17+variant); lead=28800; tail=11200; n=lead+len(pcm)+tail
  signal=np.zeros(n,dtype=np.float32); signal[lead:lead+len(pcm)]=pcm
  if variant:
   ir=np.zeros(2400,dtype=np.float32); ir[0]=1; taps=rng.integers(80,2400,size=24); ir[taps]+=rng.normal(0,.2,len(taps))*np.exp(-taps/rng.uniform(500,1400)); signal=fftconvolve(signal,ir)[:n]
  level=float(rng.uniform(.025,.25)) if variant else .12; rms=np.sqrt(np.mean(pcm**2)); signal*=level/max(.02,rms)
  noise=rng.normal(size=n).astype(np.float32); noise/=max(1e-8,np.sqrt(np.mean(noise**2))); snr=float(rng.uniform(8,30)) if variant else 45
  mixed=np.clip(signal+noise*level*10**(-snr/20),-1,1)*32767
  z=features(mixed); kf=(lead+(r['keyword_end_samples'] if positive else min(r['pcm_samples'],24000)))//160-2
  row={'split':'training' if r['voice']=='af_maple' else 'validation','label':int(positive),'index':len(rows)+len(added),'text':r['text'],'speakers':[],'path':str(src/r['pcm_path']),'samples':int(n),'keyword_end_samples':int(r['keyword_end_samples']),'sha256':r['pcm_sha256'],'engine':'kokoro-v11-fullwave','variant':variant,'lead_frame':int(lead//160),'keyword_frame':int(kf) if positive else None,'offset':offset,'frames':len(z),'voice':r['voice'],'carrier':r['carrier'],'render_variant':variant,'snr_db':snr,'room_response':bool(variant),'keyword_boundary_reference':'generated boundary-v2'}
  rows.append(row);added.append(row);chunks.append(z);offset+=len(z)
with (out/'features.bin').open('ab') as f:
 for z in chunks:f.write(z.astype('<u2').tobytes())
(out/'features.json').write_text(json.dumps(rows,indent=2)+'\n')
prov={'schema':'kokoro-v11-fullwave-overlay-v1','base':str(base),'base_features_sha256':hashlib.sha256((base/'features.bin').read_bytes()).hexdigest(),'added_records':len(added),'added_frames':sum(x['frames'] for x in added),'variants_per_record':2,'source_root':str(src),'voice_split':{'af_maple':'training','af_sol':'validation'},'acceptance_excluded':True,'threshold_selection_excluded':True}
(out/'kokoro-overlay-provenance.json').write_text(json.dumps(prov,indent=2)+'\n');print(json.dumps(prov))
