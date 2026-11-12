"""Verify original arms before rendering all-source noise/RIR controls; no model."""
import argparse,hashlib,io,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['reference','controls'],required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');pack=base/'piper-balanced-training-features-v1';source=base/'piper-balanced-training-full-v1';out=base/'balanced-rendering-factorial-v1'
assert sha(pack/'report.json')=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
assert sha(source/'manifest.jsonl')=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
r=json.loads((pack/'report.json').read_text());rows=r['records'];assert len(rows)==24192
ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
def encode(w):
 b=io.BytesIO();sf.write(b,w,16000,subtype='PCM_16',format='WAV');data=b.getvalue();b.seek(0);pcm,rate=sf.read(b,dtype='int16');assert rate==16000;return data,pcm
if a.phase=='reference':out.mkdir(exist_ok=False)
else:
 receipt=json.loads((out/'reference.json').read_text());assert receipt['verified']==16128 and receipt['all_exact'] and receipt['script_sha256']==sha(Path(__file__))
 (out/'artifacts').mkdir(exist_ok=False)
records=[]
for i in range(8064):
 d0,d2=rows[3*i],rows[3*i+2];assert d0['source']==d2['source']
 path=source/d0['source'];assert sha(path)==d0['source_sha256'];dry,rate=sf.read(path,dtype='float64');assert rate==16000
 wet=fftconvolve(dry,ir,mode='full');noise=np.random.default_rng(d2['seed']).normal(0,1,32000+len(wet)+16000);noise*=.08*10**(-15/20)/np.sqrt(np.mean(noise**2))
 arms=[('quiet',dry,None,d0),('combined',wet,noise,d2)] if a.phase=='reference' else [('noise15',dry,noise[:32000+len(dry)+16000],None),('rir',wet,None,None)]
 for arm,speech,n,original in arms:
  gain=.08/np.sqrt(np.mean(speech**2));signal=speech*gain;w=np.pad(signal,(32000,16000));pre=w.copy()
  if n is not None:w+=n
  headroom=min(1.,.95/np.max(np.abs(w)));w*=headroom
  assert np.isfinite(w).all() and np.max(np.abs(w))<1
  wav,pcm=encode(w);assert np.any(pcm) and not np.any((pcm==-32768)|(pcm==32767));raw=features(pcm)
  if original:
   assert sha(pack/original['float_path'])==original['float_sha256'];np.testing.assert_array_equal(w,np.load(pack/original['float_path']))
   assert hashlib.sha256(wav).hexdigest()==original['sha256'];assert sha(pack/original['path'])==original['sha256']
   assert sha(pack/original['raw'])==original['raw_file_sha256'];np.testing.assert_array_equal(raw,np.load(pack/original['raw']))
   assert gain==original['level_gain'] and headroom==original['gain_after_mix']
  else:
   stem=f'artifacts/{i}-{arm}';paths={}
   for name,x in [('speech',speech),('signal',signal),('padded_signal',pre),('noise',np.zeros(len(w)) if n is None else n),('wave',w),('raw',raw)]:
    q=out/(stem+'-'+name+'.npy');np.save(q,x);paths[name]={'path':str(q.relative_to(out)),'size':q.stat().st_size,'sha256':sha(q)}
   q=out/(stem+'.wav');q.write_bytes(wav);paths['wav']={'path':str(q.relative_to(out)),'size':len(wav),'sha256':sha(q)}
   e=(32000+len(signal)+159)//160-1
   records.append(dict(source_index=i,source=d0['source'],source_sha256=d0['source_sha256'],voice=d0['voice'],target=d0['target'],speed=d0['length_multiplier'],arm=arm,noise_seed=d2['seed'],noise_policy='Prefix of original combined-arm full noise vector; common scaling before prefix',level_gain=float(gain),gain_after_mix=float(headroom),signal_samples=len(signal),lead_samples=32000,tail_samples=16000,feature_crop_end=e,files=paths))
 if i%250==0:print(json.dumps(dict(phase=a.phase,completed=i+1,total=8064)),flush=True)
report=dict(phase=a.phase,verified=16128,all_exact=True,script_sha256=sha(Path(__file__)),source_report_sha256=sha(pack/'report.json'),records=records,wake_models_scored=False,final_data_used=False,limits='Pinned-library reproduction, not independent numerical implementation or acoustic truth. Noise-only truncates common combined noise vector; realized SNR can differ slightly. All sources retained; no model/training/threshold selection.')
(out/(a.phase+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
