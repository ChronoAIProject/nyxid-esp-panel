"""Precommitted complete-source speed, noise and full-tail RIR augmentation."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve,resample_poly
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
SRC=BASE/'speecht5-training-native-v1';OUT=BASE/'speecht5-training-rendered-v1'
PLAN=R/'4090-speecht5-training-render-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def versions():return {n:metadata.version(n) for n in ['numpy','scipy','soundfile']}
def prepare():
 report=R/'4090-speecht5-training-native-quality-v1.json';q=json.loads(report.read_text());assert q['generated']==q['mechanically_passed']==168 and q['pack_mechanically_qualified']
 source=SRC/'records.jsonl';rows=[json.loads(x) for x in source.read_text().splitlines()];assert len(rows)==168 and all(x['passed'] and x['role']=='training' for x in rows)
 requests=[]
 for r in rows:
  assert sha(SRC/r['path'])==r['sha256']
  for speed,up,down in [('slow',10,9),('native',1,1),('fast',10,11)]:
   for variant in [0,1,2]:
    name=r['request_id']+'-'+speed+'-v'+str(variant)
    requests.append(dict(id=name,source=r,up=up,down=down,speed=speed,speed_factor=down/up,variant=variant,noise_seed=int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')))
 p=dict(schema='speecht5-training-render-plan-v1',requests=requests,files={str(f):sha(f) for f in [Path(__file__),source,report]},versions=versions(),lead_samples=48000,tail_samples=16000,rir_delays=[0,160,480,960],rir_amplitudes=[1.,.2,-.12,.06],snr_db=[None,25,15],maximum_feature_frames=846,
  policy='All168 sources x3complete-vector speeds x3conditions =1512training-only streams. Speed resampling changes pitch as well as duration; no claim pitch-preserving. Keep complete ceil(N*up/down) speed vector, full960sample RIR convolution tail in variant2,3slead/1stail. Fixed per-request Gaussian noise seed, noise RMS over complete stream normalized relative to complete speed-transformed dry RMS. Variant0quiet;1noise25dB;2RIR+noise15dB. Predeclared postmix gain min(1,.95/peak) for noisy variants only. Native source remains untouched, no trimming/retries or score filtering. Preserve all failed rows and stop pack. All source uses remain training-only; no holdout qualification.',
  boundary='generated_container_end_sample includes complete transformed speech and RIR tail; not keyword/phoneme truth. Full command positive sources are preserved but excluded from first timing-supervised experiment due unavailable keyword end. Original timely command supervision remains in parent sampler.',training_execution_authorized=False,wake_scoring_authorized=False)
 with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(requests=len(requests),plan_sha256=sha(PLAN))),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 assert versions()==p['versions'];OUT.mkdir(exist_ok=False);impulse=np.zeros(961);impulse[p['rir_delays']]=p['rir_amplitudes'];count=0
 try:
  with (OUT/'records.jsonl').open('x') as log:
   for req in p['requests']:
    r=req['source'];source=SRC/r['path'];assert sha(source)==r['sha256'];pcm,rate=sf.read(source,dtype='int16');assert rate==16000 and len(pcm)==r['samples']
    dry=resample_poly(pcm.astype(np.float64)/32768,req['up'],req['down']);assert len(dry)==(len(pcm)*req['up']+req['down']-1)//req['down']
    speech=fftconvolve(dry,impulse,mode='full') if req['variant']==2 else dry
    values=np.pad(speech,(p['lead_samples'],p['tail_samples']));snr=p['snr_db'][req['variant']]
    if req['variant']:
     noise=np.random.default_rng(req['noise_seed']).normal(size=len(values));noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2));values+=noise
    gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if req['variant'] else 1.;values*=gain
    wav=OUT/(req['id']+'.wav');sf.write(wav,values,16000,subtype='PCM_16');saved,rate=sf.read(wav,dtype='int16')
    checks=dict(full_length=len(saved)==len(dry)+64000+(960 if req['variant']==2 else 0),format=rate==16000 and saved.ndim==1 and sf.info(wav).subtype=='PCM_16',finite_unclipped=bool(np.isfinite(values).all() and np.all(np.abs(values)<1)),no_saturation=not bool(np.any((saved==-32768)|(saved==32767))),transport=float(np.max(np.abs(saved.astype(np.float64)/32768-values)))<=1/32768+1e-7,fits_sequence=(len(saved)+159)//160-2<=846)
    row=dict(id=req['id'],source_request_id=r['request_id'],source_sha256=r['sha256'],role='training',speaker=r['speaker'],voice_id=r['voice_id'],phrase_id=r['phrase_id'],text=r['text'],label=r['label'],carrier=r['carrier'],speed=req['speed'],speed_factor=req['speed_factor'],up=req['up'],down=req['down'],variant=req['variant'],noise_seed=req['noise_seed'],snr_db=snr,gain_after_mix=gain,source_samples=len(pcm),transformed_source_samples=len(dry),samples=len(saved),lead_samples=48000,tail_samples=16000,rir_tail_samples=960 if req['variant']==2 else 0,generated_container_end_sample=48000+len(speech),native_generation_end=48000+len(dry),keyword_end=None,path=str(wav),sha256=sha(wav),pcm_sha256=hashlib.sha256(saved.astype('<i2').tobytes()).hexdigest(),checks=checks,passed=all(checks.values()))
    log.write(json.dumps(row)+'\n');log.flush();count+=1
    if not row['passed']:raise RuntimeError('Retained render failure: '+req['id'])
    if count%168==0:print(json.dumps(dict(rendered=count,total=1512)),flush=True)
  report=dict(plan_sha256=sha(PLAN),requests=count,mechanically_passed=count,whole_pack_qualified=count==1512,records_sha256=sha(OUT/'records.jsonl'),training_execution_performed=False,wake_scored=False,policy=p['policy'])
  with (R/'4090-speecht5-training-render-result-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e),completed=count))+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
