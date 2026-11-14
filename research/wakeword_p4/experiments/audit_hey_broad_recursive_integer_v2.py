"""Independent saved-file audit of complete recursive encoder/FC validation."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');p=R/'4090-hey-broad-recursive-integer-result-v2.json';r=json.loads(p.read_text());old=json.loads((R/'4090-hey-broad-equalized-screen-result-v1.json').read_text())['cases'];assert len(r['cases'])==len(old)==270
summary={}
for c,o in zip(r['cases'],old,strict=True):
 assert (c['target'],c['request_order'])==(o['target'],o['request_order'])
 f=Path(c['arrays_path']);assert hashlib.sha256(f.read_bytes()).hexdigest()==c['arrays_sha256']
 with np.load(f) as a,np.load(o['arrays_path']) as b:
  assert np.array_equal(a['actual_uint8'],b['uint8'])
  e=np.abs(a['65'].astype(np.int64)-a['actual_65'].astype(np.int64));assert int(e.max())==c['metrics_both_repeats']['65']['maximum'];assert 2*int(np.sum(e!=0))==c['metrics_both_repeats']['65']['unequal']
  assert all(len(a[k])==len(a['actual_uint8']) for k in a.files)
 assert c['all_encoder_fc_exact']==all(v['unequal']==0 for v in c['metrics_both_repeats'].values())
 s=summary.setdefault(c['target'],dict(cases=0,exact=0,actual_output_exact=0,repeat_exact=0));s['cases']+=1;s['exact']+=c['all_encoder_fc_exact'];s['actual_output_exact']+=c['actual_output_exact'];s['repeat_exact']+=c['twice_output_state_exact']
assert r['all_encoder_fc_exact']==all(c['all_encoder_fc_exact'] for c in r['cases'])
audit=dict(result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),summary=summary,checks='All270 NPZ SHA, original target/order/finalUINT8, stepcounts and saved actualFC exact-error statistics independently recomputed. Encoder internal equivalence and full repeat/statehash remain original live runner evidence, not independently recreated from saved files.',limits='Reference encoder/FC arithmetic parity only, no candidate admission, complete sigmoid kernel implementation, optimized/P4 equivalence, model-training gradient or recognition/FAR claim.')
out=R/'4090-hey-broad-recursive-integer-independent-audit-v2.json'
with out.open('x') as f:f.write(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
