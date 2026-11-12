"""Execute preregistered324 training-only augmentation attempts, retain full artifacts."""
import hashlib,importlib.metadata,io,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf
from features import features
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=Path('.local/wakeword/us-training-coverage-v1');out=Path('.local/wakeword/us-training-features-v1')
assert sha(source/'manifest.jsonl')=='350315d41263ef24d593fa46b1291ccc3ef21c95682840d65ccfcf1f1ea7369f'
audit_path=Path('research/wakeword_p4/results/4090-us-training-coverage-audit-v1.json');audit=json.loads(audit_path.read_text());assert audit['verified']==audit['clips']==108 and audit['manifest_sha256']==sha(source/'manifest.jsonl')
rows=[json.loads(x) for x in (source/'manifest.jsonl').read_text().splitlines()];assert len(rows)==108
out.mkdir(exist_ok=False);(out/'artifacts').mkdir()
ir=np.zeros(961,np.float64);ir[[0,160,480,960]]=[1,.25,-.12,.06]
records=[];xs=[];ys=[];counts=Counter()
for row in rows:
 assert row['split']=='training' and row['quality_pass'] and not row['trimmed'] and sha(source/row['path'])==row['sha256']
 original,rate=sf.read(source/row['path'],dtype='float64');assert rate==16000 and original.ndim==1
 for variant in range(3):
  seed=int(hashlib.sha256((row['path']+f':{variant}:us-training-feature-v1').encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
  speech=fftconvolve(original,ir,mode='full') if variant==2 else original.copy()
  level_gain=.08/np.sqrt(np.mean(speech**2));signal=speech*level_gain
  wave=np.pad(signal,(32000,16000));snr=None
  if variant:
   snr=25 if variant==1 else 15;noise=rng.normal(0,1,len(wave));noise*=.08*10**(-snr/20)/np.sqrt(np.mean(noise**2));wave+=noise
  gain=min(1.,.95/np.max(np.abs(wave)));wave*=gain
  assert np.isfinite(wave).all() and np.max(np.abs(wave))<1
  stem=Path(row['path']).stem+f'-v{variant}';floatpath=out/'artifacts'/(stem+'.npy');wavpath=out/'artifacts'/(stem+'.wav');rawpath=out/'artifacts'/(stem+'-raw.npy')
  np.save(floatpath,wave);sf.write(wavpath,wave,16000,subtype='PCM_16');pcm,sr=sf.read(wavpath,dtype='int16')
  buf=io.BytesIO();sf.write(buf,wave,16000,subtype='PCM_16',format='WAV');buf.seek(0);roundtrip,_=sf.read(buf,dtype='int16');np.testing.assert_array_equal(roundtrip,pcm)
  assert pcm.ndim==1 and len(pcm)==len(wave) and sr==16000 and np.any(pcm) and not np.any((pcm==-32768)|(pcm==32767))
  raw=features(pcm);assert raw.dtype==np.uint16 and raw.ndim==2 and raw.shape[1]==40;np.save(rawpath,raw)
  end=32000+len(signal);e=(end+159)//160-1;begin=(e-150)*160;stop=(e+1)*160
  fits=e>=150 and e<=len(raw) and begin<=32000 and stop>=end
  item=dict(order=len(records),source=row['path'],source_sha256=row['sha256'],source_pcm_sha256=row['pcm_sha256'],voice=row['voice'],target=row['target'],length_multiplier=row['length_multiplier'],draw=row['draw'],split='training',variant=variant,seed=seed,snr_db=snr,rir_tail_samples=960 if variant==2 else 0,source_samples=len(original),signal_samples=len(signal),level_gain=float(level_gain),gain_after_mix=float(gain),lead_samples=32000,tail_samples=16000,samples=len(pcm),float_path=str(floatpath.relative_to(out)),float_sha256=sha(floatpath),path=str(wavpath.relative_to(out)),sha256=sha(wavpath),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),raw=str(rawpath.relative_to(out)),raw_file_sha256=sha(rawpath),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),raw_shape=list(raw.shape),feature_crop_end=e,crop_start_sample=begin,crop_stop_sample=stop,entire_signal_fits=bool(fits),pack_index=None,mechanical_pass=True)
  if fits:
   crop=raw[e-150:e];assert crop.shape==(150,40);item['pack_index']=len(xs);xs.append(crop);ys.append(int(row['target']=='hey_nyx'))
  counts[f"{row['voice']}/{row['target']}/"+('included' if fits else 'unsupported_context')]+=1;records.append(item)
 print(json.dumps(dict(completed_sources=rows.index(row)+1,attempts=len(records))),flush=True)
assert len(records)==324
for voice in ['lessac','ryan','ljspeech']:
 for phrase in ['hey_nyx','hey_nick','hey_next','hey_mix']:assert counts[f'{voice}/{phrase}/included']>0
np.savez_compressed(out/'training.npz',x=np.asarray(xs,np.uint16),y=np.asarray(ys,np.float32))
report=dict(role='training-only-us-coverage-pack',source_manifest_sha256=sha(source/'manifest.jsonl'),source_audit_sha256=sha(audit_path),sources=108,attempted_variants=324,included=len(xs),unsupported_context=324-len(xs),mechanical_pass=sum(r['mechanical_pass'] for r in records),counts=dict(counts),pack_sha256=sha(out/'training.npz'),script_sha256=sha(Path(__file__)),frontend_source_sha256=sha(Path(__file__).with_name('features.py')),versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','soundfile','pymicro-features']},python=sys.version,records=records,wake_models_scored=False,final_data_used=False,limits='Complete generated source/RIR-container ledger only, not phonetic truth or physical receptive field. Same pinned conversion library roundtrip, not independent rounding implementation. Unsupported attempts retained in full, not trimmed/relabelled. No model training or automatic promotion; historical quality gaps remain.')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
