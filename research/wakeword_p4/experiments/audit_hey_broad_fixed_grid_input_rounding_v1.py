"""Independent saved-array arithmetic audit of all540 rounded-input controls."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(delta):
 five=np.asarray([sum(delta[j:j+5])/5 for j in range(len(delta)-4)])
 return dict(mean_absolute=float(np.mean(np.abs(five))),p99_absolute=float(np.quantile(np.abs(five),.99)),maximum_absolute=float(np.max(np.abs(five))))
def main():
 p=R/'4090-hey-broad-fixed-grid-input-rounding-result-v1.json';plan=R/'4090-hey-broad-fixed-grid-input-rounding-plan-v1.json';result=json.loads(p.read_text());lock=json.loads(plan.read_text());assert result['plan_sha256']==sha(plan)
 for f,h in lock['files'].items():assert sha(f)==h,f
 baseline=json.loads(Path(lock['original_result_path']).read_text());original={(c['target'],c['condition'],c['request_order']):c for c in baseline['cases']}
 seen=set();summary={}
 for c in result['cases']:
  key=(c['target'],c['condition'],c['request_order']);assert key not in seen;seen.add(key);o=original[key]
  assert sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a,np.load(o['arrays_path'],allow_pickle=False) as b:
   rounded=a['rounded_input_float'];assert np.isfinite(rounded).all() and np.array_equal(rounded,a['repeat']) and c['twice_reset_exact']
   assert np.array_equal(a['sample_end'],b['sample_end'])
   os,oz=o['output_quantization'];integer=(b['uint8'].astype(np.float64)-oz)*os
   deltas=dict(input_rounding=rounded-b['float'],remaining_quantization=integer-rounded,total_original=integer-b['float'])
   assert np.array_equal(deltas['input_rounding'],a['input_rounding_delta']) and np.array_equal(deltas['remaining_quantization'],a['remaining_delta'])
   assert np.allclose(deltas['input_rounding']+deltas['remaining_quantization'],deltas['total_original'],rtol=0,atol=1e-15)
   for kind,delta in deltas.items():
    computed=stat(delta)
    for name,value in computed.items():assert np.isclose(value,c[kind][name],rtol=0,atol=1e-14),(key,kind,name)
  name=c['condition']+'/'+c['target'];s=summary.setdefault(name,dict(cases=0,original_failed=0,original_failed_input_p99_above_point1=0,original_failed_remaining_p99_above_point1=0,worst_input_p99=0,worst_remaining_p99=0,worst_input_max=0,worst_remaining_max=0));s['cases']+=1;s['original_failed']+=not o['passed'];s['original_failed_input_p99_above_point1']+=bool(not o['passed'] and c['input_rounding']['p99_absolute']>.1);s['original_failed_remaining_p99_above_point1']+=bool(not o['passed'] and c['remaining_quantization']['p99_absolute']>.1)
  for kind,short in [('input_rounding','input'),('remaining_quantization','remaining')]:
   for metric,label in [('p99_absolute','p99'),('maximum_absolute','max')]:s['worst_'+short+'_'+label]=max(s['worst_'+short+'_'+label],c[kind][metric])
 assert seen==set(original) and len(seen)==540 and len(summary)==4 and all(s['cases']==135 for s in summary.values())
 out=R/'4090-hey-broad-fixed-grid-input-rounding-independent-audit-v1.json'
 with out.open('x') as f:json.dump(dict(result_sha256=sha(p),summary=summary,limits='Independent all540 fileSHA/identities/repeat/clock/signed-delta decomposition and five stats; no independent inference replay. Descriptive nonlinear comparison; neither component is an independent causal effect. No altered gate, newdev, recognition, admission or promotion.'),f,indent=2);f.write('\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
