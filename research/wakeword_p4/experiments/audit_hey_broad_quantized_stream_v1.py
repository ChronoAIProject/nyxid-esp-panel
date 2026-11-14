import json,hashlib
from pathlib import Path
import numpy as np
root=Path('research/wakeword_p4/results');r=json.loads((root/'4090-hey-broad-quantized-stream-result-v1.json').read_text());old=json.loads((root/'4090-hey-broad-equalized-screen-result-v1.json').read_text())['cases'];assert len(r['cases'])==len(old)==270
summary={}
for c,o in zip(r['cases'],old,strict=True):
 assert (c['target'],c['request_order'])==(o['target'],o['request_order'])
 p=Path(c['arrays_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==c['arrays_sha256']
 with np.load(p,allow_pickle=False) as a,np.load(o['arrays_path'],allow_pickle=False) as b:
  assert np.array_equal(a['probability'],a['repeat'][:,1]) and np.array_equal(a['logit'],a['repeat'][:,0])
  for k in ['sample_end','support_unclamped']:assert np.array_equal(a[k],b[k])
  s,z=o['output_quantization'];assert np.array_equal(a['actual_int8'],(b['uint8'].astype(np.float64)-z)*s)
  e=np.abs(np.convolve(a['probability']-a['actual_int8'],np.ones(5)/5,'valid'))
  stats=dict(mean=float(np.mean(e)),p99=float(np.quantile(e,.99)),maximum=float(np.max(e)))
  assert stats==c['actual_int8_smoothed_abs_error'],(stats,c)
  pe=float(np.max(np.abs(a['offline_phase']-a['probability'][74:])));le=float(np.max(np.abs(a['offline_logit_phase']-a['logit'][74:])))
  assert pe==c['phase_probability_max_error'] and le==c['phase_logit_max_error']
  assert c['phase_passed']==(pe<=1e-4)
  assert c['runtime_fidelity_passed']==(stats['mean']<=.02 and stats['p99']<=.1 and stats['maximum']<=.25)
 t=summary.setdefault(c['target'],dict(cases=0,phase_pass=0,fidelity_pass=0,exact_repeat=0,worst_mean=0,worst_p99=0,worst_maximum=0,failures=[]))
 t['cases']+=1;t['phase_pass']+=c['phase_passed'];t['fidelity_pass']+=c['runtime_fidelity_passed'];t['exact_repeat']+=c['twice_output_state_exact']
 for k,v in stats.items():t['worst_'+k]=max(t['worst_'+k],v)
 if not c['phase_passed'] or not c['runtime_fidelity_passed']:t['failures'].append(c['request_order'])
assert r['all_phase_passed']==all(c['phase_passed'] for c in r['cases'])
assert r['all_runtime_fidelity_passed']==all(c['runtime_fidelity_passed'] for c in r['cases'])
audit=dict(schema='hey-broad-quantized-stream-independent-audit-v1',result_sha256=hashlib.sha256((root/'4090-hey-broad-quantized-stream-result-v1.json').read_bytes()).hexdigest(),summary=summary,checks='All270 archiveSHA, original clocks, actualINT8, repeat outputs, offline probability/rawlogit phase and original five error statistics independently recomputed without importing runner or TF.',limits='Internal state exact-repeat hashes are runner evidence; saved NPZ contains outputs not full state bytes. Surrogate fidelity does not establish candidate numerical admission, recognition or QAT training success.')
(root/'4090-hey-broad-quantized-stream-independent-audit-v1.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
