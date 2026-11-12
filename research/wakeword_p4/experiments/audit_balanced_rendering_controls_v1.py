"""Independent all-control array/PCM/frontend/container replay, no wake model."""
import hashlib,io,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
base=Path('.local/wakeword');root=base/'balanced-rendering-factorial-v1';pack=base/'piper-balanced-training-features-v1';source=base/'piper-balanced-training-full-v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((root/'controls.json').read_text());ref=json.loads((root/'reference.json').read_text());assert r['script_sha256']==ref['script_sha256']==sha(Path(__file__).with_name('balanced_rendering_factorial_v1.py'));assert ref['all_exact'] and ref['verified']==16128
assert sha(pack/'report.json')==r['source_report_sha256']=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
old=json.loads((pack/'report.json').read_text())['records'];assert len(r['records'])==16128 and not r['wake_models_scored'] and not r['final_data_used']
ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06];snrs=[];gains=[];seen=set();quality=0
for j,d in enumerate(r['records']):
 i=j//2;arm=['noise15','rir'][j%2];assert d['source_index']==i and d['arm']==arm and (i,arm) not in seen;seen.add((i,arm));a,b=old[3*i],old[3*i+2]
 assert d['source']==a['source'] and d['source_sha256']==a['source_sha256'] and d['voice']==a['voice'] and d['target']==a['target'] and d['speed']==a['length_multiplier'] and d['noise_seed']==b['seed']
 p=source/a['source'];assert sha(p)==a['source_sha256'];dry,rate=sf.read(p,dtype='float64');assert rate==16000 and dry.ndim==1
 wet=fftconvolve(dry,ir,mode='full');speech=dry if arm=='noise15' else wet;level=.08/np.sqrt(np.mean(speech**2));signal=speech*level;padded=np.pad(signal,(32000,16000))
 if arm=='noise15':
  n=np.random.default_rng(b['seed']).normal(0,1,32000+len(wet)+16000);n*=.08*10**(-15/20)/np.sqrt(np.mean(n**2));n=n[:len(padded)];snrs.append(float(20*np.log10(.08/np.sqrt(np.mean(n**2)))))
 else:n=np.zeros(len(padded))
 wave=padded+n;gain=min(1.,.95/np.max(np.abs(wave)));wave*=gain;gains.append(float(gain))
 expected=dict(speech=speech,signal=signal,padded_signal=padded,noise=n,wave=wave)
 for key,entry in d['files'].items():
  path=root/entry['path'];assert path.stat().st_size==entry['size'] and sha(path)==entry['sha256']
  if key in expected:np.testing.assert_array_equal(np.load(path,allow_pickle=False),expected[key])
 wav=io.BytesIO();sf.write(wav,wave,16000,subtype='PCM_16',format='WAV');assert wav.getvalue()==(root/d['files']['wav']['path']).read_bytes();wav.seek(0);pcm,sr=sf.read(wav,dtype='int16')
 assert sr==16000 and np.any(pcm) and not np.any((pcm==-32768)|(pcm==32767)) and np.isfinite(wave).all() and np.max(np.abs(wave))<1
 raw=features(pcm);np.testing.assert_array_equal(raw,np.load(root/d['files']['raw']['path']));assert raw.dtype==np.uint16 and raw.shape[1]==40
 e=(32000+len(signal)+159)//160-1;assert e==d['feature_crop_end'] and d['lead_samples']==32000 and d['tail_samples']==16000 and len(signal)==d['signal_samples'] and len(pcm)==32000+len(signal)+16000
 assert e>=150 and e<=len(raw) and (e-150)*160<=32000 and (e+1)*160>=32000+len(signal);assert level==d['level_gain'] and gain==d['gain_after_mix'];quality+=1
 if j%1000==0:print(json.dumps(dict(verified=j+1,total=16128)),flush=True)
assert quality==16128 and len(seen)==16128
out=dict(verified=quality,all_arrays_pcm_frontend_exact=True,all_complete_containers=True,noise_only_realized_snr_db_min=min(snrs),noise_only_realized_snr_db_max=max(snrs),headroom_gain_min=min(gains),headroom_gain_max=max(gains),controls_report_sha256=sha(root/'controls.json'),reference_report_sha256=sha(root/'reference.json'),script_sha256=sha(Path(__file__)),wake_models_scored=False,final_data_used=False,limits='Independent replay code shares pinned scipy/soundfile/frontend libraries. Original noise prefix finite-length SNR disclosed. Complete process containers not phonetic truth; no inference/training/threshold selection.')
Path('research/wakeword_p4/results/4090-balanced-rendering-controls-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
