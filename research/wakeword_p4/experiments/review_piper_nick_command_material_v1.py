"""Independent complete native/duration/resample/PCM and source-pair audit."""
import hashlib,json,os
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/piper-nick-command-material-v1')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 planpath=R/'4090-piper-nick-command-material-plan-v1.json';plan=read(planpath)
 receipt=R/'4090-piper-nick-command-material-execution-v1.json';assert read(receipt)['actual_exit_code']==0
 reportpath=R/'4090-piper-nick-command-material-original-v1.json';report=read(reportpath)
 assert report['plan_sha256']==sha(planpath) and report['cells']==6048
 for path,h in plan['files'].items():assert sha(path)==h,path
 ledger=B/'records.jsonl';assert sha(ledger)==report['records_sha256']
 rows=[json.loads(s) for s in ledger.read_text().splitlines()];assert len(rows)==len(plan['rows'])==6048
 failures=[];identities=set();bytes_verified=0
 for i,(row,request) in enumerate(zip(rows,plan['rows'],strict=True)):
  assert row['order']==request['order']==i and row['request']==request
  original=request['original_request'];anchor=request['positive_anchor']
  assert original['split']=='training' and original['target']=='hey_nyx'
  assert request['text']==original['text'].replace('Hey Nicks,','Hey Nick,',1)
  assert request['seed']==original['seed'] and anchor['text']==original['text']
  assert str(original['speaker_id'])==anchor['pair_key']['conditioning'] and original['length_scale']==anchor['pair_key']['scale'] and original['carrier']==anchor['pair_key']['carrier']
  identity=(original['speaker_id'],original['length_scale'],original['carrier']);assert identity not in identities;identities.add(identity)
  assert sha(anchor['source_path'])==anchor['source_sha256']
  files=[B/f'{i:05}.npz',B/f'{i:05}.wav'];assert set(row['files'])==set(map(str,files))
  for p in files:
   spec=row['files'][str(p)];assert p.stat().st_size==spec['bytes'] and sha(p)==spec['sha256'];bytes_verified+=spec['bytes']
  with np.load(files[0],allow_pickle=False) as z:
   assert set(z.files)=={'native','durations','resampled'}
   native=z['native'];dur=z['durations'];resampled=z['resampled']
   assert native.ndim==dur.ndim==resampled.ndim==1
   assert np.array_equal(resample_poly(native,320,441),resampled)
   pcm,rate=sf.read(files[1],dtype='int16');info=sf.info(files[1])
   # libsndfile PCM16 conversion floors and clips fullscale; test every sample.
   expected=np.clip(np.floor(resampled.astype(np.float64)*32768),-32768,32767).astype(np.int16)
   assert np.array_equal(pcm,expected)
   rms=float(np.sqrt(np.mean(resampled.astype(np.float64)**2)))
   checks=dict(finite=bool(np.isfinite(native).all() and np.isfinite(dur).all() and np.isfinite(resampled).all()),format=rate==16000 and pcm.ndim==1 and info.subtype=='PCM_16',duration=.2<=len(pcm)/16000<=10,rms=rms>=.001,native_no_fullscale=bool(np.all(np.abs(native)<1)),resampled_no_fullscale=bool(np.all(np.abs(resampled)<1)),pcm_no_endpoint=not bool(np.any((pcm==-32768)|(pcm==32767))),full_length=len(pcm)==len(resampled)==(len(native)*320+440)//441,durations_integer=bool(np.isfinite(dur).all() and np.all(dur>=0) and np.array_equal(dur,np.rint(dur))),token_count=len(dur)==len(plan['text_tokens'][request['text']]),native_duration=float(np.sum(dur,dtype=np.float64))==len(native))
   assert checks==row['checks'] and all(checks.values())==row['quality_pass'] and rms==row['rms'] and len(pcm)==row['samples']
   assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
   if not all(checks.values()):failures.append(dict(order=i,checks=checks))
  if (i+1)%1000==0:print(json.dumps(dict(verified=i+1)),flush=True)
 assert failures==report['failures'] and (not failures)==report['all_mechanical_checks_passed']
 out=dict(schema='piper-nick-command-material-independent-review-v1',all_audit_checks_passed=True,material_admitted=not failures,cells_verified=6048,files_verified=12096,bytes_verified=bytes_verified,conditioning_ids=len({x[0] for x in identities}),failures=failures,plan_sha256=sha(planpath),report_sha256=sha(reportpath),receipt_sha256=sha(receipt),reviewer_sha256=sha(__file__),training_enabled=False,limits='Complete saved byte/pair/duration/resample/PCM audit; no wake scoring or audible pronunciation guarantee, no independent human voices, no model training or promotion.')
 with (R/'4090-piper-nick-command-material-review-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
