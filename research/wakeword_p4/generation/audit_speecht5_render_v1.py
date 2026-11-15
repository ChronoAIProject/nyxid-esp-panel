"""Reconstruct full render with sparse direct RIR accumulation, no wake model."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=R/'4090-speecht5-training-render-plan-v1.json';p=json.loads(plan.read_text());records=BASE/'speecht5-training-rendered-v1/records.jsonl';rows=[json.loads(x) for x in records.read_text().splitlines()];assert len(rows)==len(p['requests'])==1512;maximum=0.
 for row,req in zip(rows,p['requests'],strict=True):
  assert row['id']==req['id'] and row['role']=='training'
  source=BASE/'speecht5-training-native-v1'/req['source']['path'];assert sha(source)==req['source']['sha256']
  pcm,rate=sf.read(source,dtype='int16');assert rate==16000
  dry=resample_poly(pcm.astype(np.float64)/32768,req['up'],req['down'])
  if req['variant']==2:
   speech=np.zeros(len(dry)+960)
   for delay,amplitude in zip(p['rir_delays'],p['rir_amplitudes'],strict=True):speech[delay:delay+len(dry)]+=amplitude*dry
  else:speech=dry
  expected=np.pad(speech,(48000,16000))
  if req['variant']:
   noise=np.random.default_rng(req['noise_seed']).normal(size=len(expected));noise*=np.sqrt(np.mean(dry**2))*10**(-p['snr_db'][req['variant']]/20)/np.sqrt(np.mean(noise**2));expected+=noise
  gain=min(1.,.95/max(float(np.max(np.abs(expected))),1e-8)) if req['variant'] else 1.;expected*=gain
  path=Path(row['path']);assert sha(path)==row['sha256'];actual,rate=sf.read(path,dtype='int16');assert rate==16000 and len(actual)==len(expected)==row['samples']
  assert row['native_generation_end']==48000+len(dry) and row['generated_container_end_sample']==48000+len(speech) and row['rir_tail_samples']==len(speech)-len(dry)
  error=float(np.max(np.abs(actual.astype(np.float64)/32768-expected)));assert error<=1/32768+1e-7 and np.isfinite(expected).all() and np.all(np.abs(expected)<1)
  assert not np.any((actual==-32768)|(actual==32767));maximum=max(maximum,error)
 result=dict(schema='speecht5-render-independent-integrity-audit-v1',rows=1512,all_passed=True,plan_sha256=sha(plan),records_sha256=sha(records),audit_script_sha256=sha(__file__),rir_method='Direct sparse delayed accumulation, independent of FFT renderer',maximum_transport_error=maximum,all_source_and_rir_tail_samples_preserved=True,training_only=True,wake_scored=False,limits='Mechanical augmentation integrity only; no phonetic or recognition correctness. Speed transform changes pitch and duration; all native source bytes remain preserved.')
 with (R/'4090-speecht5-training-render-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 (R/'4090-speecht5-training-render-records-v1.jsonl.gz').write_bytes(gzip.compress(records.read_bytes(),mtime=0));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
