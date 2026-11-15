"""Full development render reconstruction with sparse direct RIR, no scores."""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 plan=R/'4090-melo-development-render-plan-v1.json';p=json.loads(plan.read_text());record_path=BASE/'melo-development-rendered-v1/records.jsonl';rows=[json.loads(x) for x in record_path.read_text().splitlines()];sources=[json.loads(x) for x in (BASE/'melo-development-native-v1/records.jsonl').read_text().splitlines()];assert len(rows)==len(p['requests'])==432;maximum=0.
 for row,req in zip(rows,p['requests'],strict=True):
  source=sources[req['source_index']];assert row['id']==req['id'] and row['source_id']==source['request_id'] and row['role']=='unconsumed-threshold-development'
  path=BASE/'melo-development-native-v1'/source['path'];assert sha(path)==source['sha256'];pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and len(pcm)==row['source_samples']
  dry=pcm.astype(np.float64)/32768
  if req['variant']==2:
   speech=np.zeros(len(dry)+1232)
   for delay,amplitude in zip(p['rir_delays'],p['rir_amplitudes'],strict=True):speech[delay:delay+len(dry)]+=amplitude*dry
  else:speech=dry
  expected=np.pad(speech,(48000,16000))
  if req['variant']:
   noise=np.random.default_rng(req['noise_seed']).normal(size=len(expected));noise*=np.sqrt(np.mean(dry**2))*10**(-p['snr_db'][req['variant']]/20)/np.sqrt(np.mean(noise**2));expected+=noise
  gain=min(1.,.95/max(float(np.max(np.abs(expected))),1e-8)) if req['variant'] else 1.;expected*=gain
  wav=Path(row['path']);assert sha(wav)==row['sha256'];actual,rate=sf.read(wav,dtype='int16');assert rate==16000 and len(actual)==len(expected)==row['samples'];assert hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
  last=2*source['input_frontend']['phones'].index('.')+2;native=sum(source['generated_conditioning_duration_samples'][:last+1]);prefix=(native*16000+44099)//44100;tail=len(speech)-len(dry)
  assert row['generated_prefix_end_sample']==48000+prefix and row['full_rir_prefix_end_sample']==48000+prefix+tail
  assert row['generated_container_end_sample']==48000+len(dry) and row['full_rir_container_end_sample']==48000+len(speech) and row['rir_tail_samples']==tail
  error=float(np.max(np.abs(actual.astype(np.float64)/32768-expected)));assert error<=1/32768+1e-7 and np.isfinite(expected).all() and np.all(np.abs(expected)<1) and not np.any((actual==-32768)|(actual==32767));maximum=max(maximum,error)
 result=dict(schema='melo-development-render-integrity-v1',rows=432,all_passed=True,script_sha256=sha(__file__),plan_sha256=sha(plan),records_sha256=sha(record_path),rir_method='Independent sparse delayed accumulation',maximum_transport_error=maximum,all_sources_and_rir_tails_preserved=True,synthetic_clocks_verified=True,phonetic_truth_claimed=False,wake_scored=False,acceptance_accessed=False,limits='Mechanical waveform and conditioning-clock integrity only. Not standalone source-role qualification or scoring authorization; all acceptance voices remain ungenerated.')
 with (R/'4090-melo-development-render-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 (R/'4090-melo-development-render-records-v1.jsonl.gz').write_bytes(gzip.compress(record_path.read_bytes(),mtime=0));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
