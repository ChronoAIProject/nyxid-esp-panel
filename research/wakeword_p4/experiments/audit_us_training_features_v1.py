"""Replay every augmentation/PCM/frontend/crop in memory; no wake model."""
import hashlib,io,json,sys
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/us-training-features-v1');source=Path('.local/wakeword/us-training-coverage-v1');r=json.loads((root/'report.json').read_text())
assert r['attempted_variants']==r['mechanical_pass']==324 and not r['wake_models_scored'] and not r['final_data_used']
assert sha(root/'training.npz')==r['pack_sha256']=='4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da'
rows=[json.loads(x) for x in (source/'manifest.jsonl').read_text().splitlines()];assert len(rows)==108
assert sha(source/'manifest.jsonl')==r['source_manifest_sha256']
ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
with np.load(root/'training.npz',allow_pickle=False) as z:
 assert z['x'].dtype==np.uint16 and z['x'].shape==(324,150,40) and z['y'].dtype==np.float32 and z['y'].shape==(324,)
 for order,d in enumerate(r['records']):
  row=rows[order//3];v=order%3
  assert d['order']==order and d['variant']==v and d['source']==row['path'] and d['source_sha256']==row['sha256'] and d['split']=='training'
  assert sha(source/row['path'])==row['sha256']
  pcm,sr=sf.read(source/row['path'],dtype='float64');assert sr==16000
  signal=fftconvolve(pcm,ir,mode='full') if v==2 else pcm.copy();level=.08/np.sqrt(np.mean(signal**2));signal*=level
  wave=np.pad(signal,(32000,16000));seed=int(hashlib.sha256((row['path']+f':{v}:us-training-feature-v1').encode()).hexdigest()[:16],16);assert seed==d['seed']
  if v:
   rng=np.random.default_rng(seed);noise=rng.normal(0,1,len(wave));noise*=.08*10**(-(25 if v==1 else 15)/20)/np.sqrt(np.mean(noise**2));wave+=noise
  gain=min(1.,.95/np.max(np.abs(wave)));wave*=gain
  assert gain==d['gain_after_mix'] and level==d['level_gain']
  for pathkey,hashkey in [('float_path','float_sha256'),('path','sha256'),('raw','raw_file_sha256')]:assert sha(root/d[pathkey])==d[hashkey]
  saved=np.load(root/d['float_path'],allow_pickle=False);assert saved.dtype==np.float64;np.testing.assert_array_equal(wave,saved)
  buf=io.BytesIO();sf.write(buf,wave,16000,subtype='PCM_16',format='WAV');buf.seek(0);quant,_=sf.read(buf,dtype='int16');actual,rate=sf.read(root/d['path'],dtype='int16');np.testing.assert_array_equal(actual,quant)
  assert rate==16000 and actual.ndim==1 and len(actual)==d['samples']==32000+len(signal)+16000 and np.any(actual) and np.isfinite(wave).all() and np.max(np.abs(wave))<1 and not np.any((actual==-32768)|(actual==32767))
  assert hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest()==d['pcm_sha256']
  raw=features(actual);savedraw=np.load(root/d['raw'],allow_pickle=False);np.testing.assert_array_equal(raw,savedraw);assert raw.dtype==np.uint16 and list(raw.shape)==d['raw_shape'] and hashlib.sha256(raw.tobytes()).hexdigest()==d['raw_sha256']
  end=32000+len(signal);e=(end+159)//160-1;assert e==d['feature_crop_end'] and e>=150 and e<=len(raw) and (e-150)*160<=32000 and (e+1)*160>=end
  assert d['entire_signal_fits'] and d['pack_index']==order;np.testing.assert_array_equal(z['x'][order],raw[e-150:e]);assert z['y'][order]==int(row['target']=='hey_nyx')
assert len(r['records'])==324
out=dict(scope='Independent read-only324-attempt full augmentation, pinned-converter, frontend and crop replay',verified=324,unsupported_context=0,pack_sha256=r['pack_sha256'],original_report_sha256=sha(root/'report.json'),script_sha256=sha(Path(__file__)),all_arrays_pcm_and_frontend_exact=True,limits='Shares pinned numerical/conversion/frontend libraries, not independent implementations. Complete-container ledger only, no phonetic truth, model scoring/training or new-development/final access.')
Path('research/wakeword_p4/results/4090-us-training-features-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
