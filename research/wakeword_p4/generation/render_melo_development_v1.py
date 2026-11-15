"""Complete development sources and fixed synthetic clocks; no detector call."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
SRC=BASE/'melo-development-native-v1';OUT=BASE/'melo-development-rendered-v1'
PLAN=R/'4090-melo-development-render-plan-v1.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def versions():return {n:metadata.version(n) for n in ['numpy','scipy','soundfile']}
def prepare():
 quality=R/'4090-melo-development-native-quality-v1.json';q=json.loads(quality.read_text());assert q['whole_pack_mechanically_qualified'] and q['generated']==q['mechanically_passed']==144 and not q['wake_scoring_performed']
 records=SRC/'records.jsonl';rows=[json.loads(x) for x in records.read_text().splitlines()];requests=[]
 for index,row in enumerate(rows):
  assert row['role']=='development' and row['passed']
  for variant in [0,1,2]:
   environment=f"{row['voice_id']}:{row['speed']}:{row['carrier']}:{variant}:melo-dev-render-v1"
   requests.append(dict(id=row['request_id']+'-v'+str(variant),source_index=index,source_id=row['request_id'],variant=variant,noise_seed=int.from_bytes(hashlib.sha256(environment.encode()).digest()[:4],'little')))
 p=dict(schema='melo-development-render-plan-v1',files={str(f):sha(f) for f in [Path(__file__),records,quality,R/'4090-melo-prospective-request-plan-v1.json']},versions=versions(),requests=requests,lead_samples=48000,tail_samples=16000,rir_delays=[0,208,624,1232],rir_amplitudes=[1.,-.18,.1,.04],snr_db=[None,25,15],
  policy='All144 complete native16kPCM sources x3fixed conditions=432development streams. Native engine speed already fixed, no further time transform. Full source +full1232sample RIRtail +3slead/1stail. Gaussian noise seed excludes phrase so pairedtargets share environment seed; different length/RMS means different realized normalization, not identical noise magnitude. noisy gain=min(1,.95/peak),quiet gain1. No trimming/regeneration/score filtering; any failure retains output and stops pack. Shared parent and unknown physical/pretraining overlap remain disclosed. No training or acceptance waveforms.',
  synthetic_prefix_clock='First period in authenticated original frontend phone list, include its immediately following inserted blank. Expanded index=2*period_index+2 inclusive. Sum saved internal token duration samples through that index,ceil44100to16000; add lead and optionally fullRIRtail for respective clocks. Conditioning/punctuation boundary only, not observed final phoneme or human latency truth. Positive hit remains ANY post-source appwake as frozen selector rule; clocks descriptive, not extra automatic failure.',
  whole_clock='Full saved source end and full source+RIRtail end. Do not crop sources at internal prefix boundary.',scoring_authorized=False,acceptance_accessed=False)
 with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(requests=len(requests),plan_sha256=sha(PLAN))),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 assert versions()==p['versions'];OUT.mkdir(exist_ok=False);sources=[json.loads(x) for x in (SRC/'records.jsonl').read_text().splitlines()];impulse=np.zeros(1233);impulse[p['rir_delays']]=p['rir_amplitudes'];count=0
 try:
  with (OUT/'records.jsonl').open('x') as log:
   for req in p['requests']:
    r=sources[req['source_index']];assert req['source_id']==r['request_id'];source=SRC/r['path'];assert sha(source)==r['sha256'];pcm,rate=sf.read(source,dtype='int16');assert rate==16000 and len(pcm)==r['samples']
    end_index=2*r['input_frontend']['phones'].index('.')+2;native_prefix=sum(r['generated_conditioning_duration_samples'][:end_index+1]);prefix=(native_prefix*16000+44099)//44100
    assert 0<prefix<=len(pcm)
    dry=pcm.astype(np.float64)/32768;speech=fftconvolve(dry,impulse,mode='full') if req['variant']==2 else dry;values=np.pad(speech,(48000,16000));snr=p['snr_db'][req['variant']]
    if req['variant']:
     noise=np.random.default_rng(req['noise_seed']).normal(size=len(values));noise*=np.sqrt(np.mean(dry**2))*10**(-snr/20)/np.sqrt(np.mean(noise**2));values+=noise
    gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if req['variant'] else 1.;values*=gain
    path=OUT/(req['id']+'.wav');sf.write(path,values,16000,subtype='PCM_16');saved,rate=sf.read(path,dtype='int16');tail=1232 if req['variant']==2 else 0
    checks=dict(full_source_tail=len(saved)==len(pcm)+64000+tail,format=rate==16000 and saved.ndim==1 and sf.info(path).subtype=='PCM_16',finite_unclipped=bool(np.isfinite(values).all() and np.all(np.abs(values)<1)),no_saturation=not bool(np.any((saved==-32768)|(saved==32767))),transport=float(np.max(np.abs(saved.astype(np.float64)/32768-values)))<=1/32768+1e-7)
    row=dict(id=req['id'],role='unconsumed-threshold-development',source_id=r['request_id'],voice_id=r['voice_id'],speed=r['speed'],carrier=r['carrier'],phrase_id=r['phrase_id'],text=r['text'],variant=req['variant'],noise_seed=req['noise_seed'],snr_db=snr,gain_after_mix=gain,source_sha256=r['sha256'],source_samples=len(pcm),samples=len(saved),lead_samples=48000,tail_samples=16000,rir_tail_samples=tail,prefix_conditioning_last_index=end_index,native_prefix_conditioning_sample_44100=native_prefix,generated_prefix_end_sample=48000+prefix,full_rir_prefix_end_sample=48000+prefix+tail,generated_container_end_sample=48000+len(pcm),full_rir_container_end_sample=48000+len(speech),path=str(path),sha256=sha(path),pcm_sha256=hashlib.sha256(saved.astype('<i2').tobytes()).hexdigest(),checks=checks,passed=all(checks.values()))
    log.write(json.dumps(row)+'\n');log.flush();count+=1
    if not row['passed']:raise RuntimeError('Retained render failure: '+req['id'])
  result=dict(plan_sha256=sha(PLAN),requests=count,mechanically_passed=count,whole_pack_qualified=count==432,records_sha256=sha(OUT/'records.jsonl'),wake_scored=False,acceptance_accessed=False,policy=p['policy'],synthetic_prefix_clock=p['synthetic_prefix_clock'])
  with (R/'4090-melo-development-render-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result),flush=True)
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(type=type(e).__name__,error=str(e),completed=count))+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
