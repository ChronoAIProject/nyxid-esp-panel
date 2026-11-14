"""Independent saved-array checks for the prospective internal FP32 fullstream run."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/fp32-internal-fullstream-v1')
def check(a,samples):
 required={'raw','input','tf','tf_repeat','litert','litert_repeat','offline_phase','offline_nominal','support_unclamped','sample_end'}
 assert set(a)==required
 raw=a['raw'];assert raw.ndim==2 and raw.shape[1]==40
 values=np.clip(np.pad(raw,((0,(-len(raw))%3),(0,0))).astype(np.float32)/25.6,0,26)
 assert a['input'].dtype==np.float32 and a['input'].shape==values.shape and a['input'].tobytes()==values.tobytes()
 steps=len(values)//3;assert steps>=75
 for k in ['tf','tf_repeat','litert','litert_repeat']:
  assert a[k].shape==(steps,) and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
 for k in ['offline_phase','offline_nominal']:
  assert a[k].shape==(steps-74,) and a[k].dtype==np.float32 and np.isfinite(a[k]).all()
 for k in ['tf','litert']:assert a[k].tobytes()==a[k+'_repeat'].tobytes()
 clock=(3*np.arange(1,steps+1)+2)*160
 for k,want in [('support_unclamped',clock),('sample_end',np.minimum(clock,samples))]:assert np.array_equal(a[k],want)
 gap=float(np.max(np.abs(a['tf'].astype(float)-a['litert'].astype(float))));phase=float(np.max(np.abs(a['offline_phase'].astype(float)-a['tf'][74:].astype(float))))
 return dict(max_abs=gap,phase_max_abs=phase,passed=max(gap,phase)<=1e-4)
def run():
 lp=R/'4090-fp32-internal-fullstream-execution-v1.json';p=json.loads(lp.read_text());result=json.loads((ROOT/'result.json').read_text());assert result['execution_sha256']==hashlib.sha256(lp.read_bytes()).hexdigest();assert len(result['cases'])==540
 seen=set();rows=[]
 for c in result['cases']:
  key=(c['target'],c['condition'],c['index']);assert key not in seen;seen.add(key);request=p['requests'][c['index']];assert request['order']==c['request_order'];path=Path(c['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['sha256']
  with np.load(path,allow_pickle=False) as arrays:d=check({k:arrays[k] for k in arrays.files},request['samples'])
  assert all(d[k]==c[k] for k in d) and c['finite'] and c['repeat_exact'];rows.append(dict(key=key,**d))
 expected={(m['target'],m['condition'],i) for m in p['models'] for i in range(135)};assert seen==expected
 (R/'4090-fp32-fullstream-independent-arrays-v1.json').write_text(json.dumps(dict(cases=rows,all_passed=all(r['passed'] for r in rows),limits='Saved arrays only. Original PCM/frontend computation and actual MicroInterpreter still need separate evidence; no model invocation.'),indent=2)+'\n')
def constructed():
 raw=np.zeros((226,40),np.uint16);steps=76;clock=(3*np.arange(1,steps+1)+2)*160;values=np.pad(raw,((0,2),(0,0))).astype(np.float32)/25.6
 a=dict(raw=raw,input=values,tf=np.zeros(steps,np.float32),tf_repeat=np.zeros(steps,np.float32),litert=np.zeros(steps,np.float32),litert_repeat=np.zeros(steps,np.float32),offline_phase=np.zeros(2,np.float32),offline_nominal=np.zeros(2,np.float32),support_unclamped=clock,sample_end=np.minimum(clock,36000));assert check(a,36000)['passed'];checks=[]
 for k in ['input','tf_repeat','offline_phase','support_unclamped','sample_end']:
  b={n:v.copy() for n,v in a.items()};b[k].flat[0]+=1
  try:d=check(b,36000)
  except AssertionError:checks.append(k);continue
  assert not d['passed'];checks.append(k)
 b={n:v.copy() for n,v in a.items()};b['litert'][0]=b['litert_repeat'][0]=.001;assert not check(b,36000)['passed'];checks.append('startup_error_retained')
 b={n:v.copy() for n,v in a.items()};b['tf'][0]=np.nan
 try:check(b,36000)
 except AssertionError:checks.append('nonfinite')
 assert len(checks)==7
 (R/'4090-fp32-fullstream-array-constructed-checks-v1.json').write_text(json.dumps(dict(positive=1,negative_checks=checks,all_passed=True,actual_fullstream_read=False,limits='Constructed numerical fixtures only, no actual model/PCM evidence.'),indent=2)+'\n');print(checks)
if __name__=='__main__':
 import sys
 {'run':run,'constructed':constructed}[sys.argv[1]]()
