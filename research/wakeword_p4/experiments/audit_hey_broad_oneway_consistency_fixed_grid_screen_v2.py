"""Independent full540 numerical reduction, unchanged prospective gates."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-result-v1.json';result=json.loads(p.read_text());execution=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-execution-v1.json';assert result['execution_sha256']==sha(execution);lock=json.loads(execution.read_text())
 for f,h in lock['files'].items():assert sha(f)==h,f
 original=json.loads((R/'4090-hey-broad-trained-quantization-plan-v1.json').read_text());requests={r['order']:r for r in original['requests']};g=original['numerical_gates'];assert len(result['cases'])==540
 summaries={};seen=set()
 for case in result['cases']:
  key=(case['target'],case['condition'],case['request_order']);assert key not in seen;seen.add(key);f=Path(case['arrays_path']);assert sha(f)==case['arrays_sha256'];row=requests[case['request_order']]
  with np.load(f,allow_pickle=False) as a:
   floating=a['float'];integer=a['uint8'];repeat=np.array_equal(floating,a['float_repeat']) and np.array_equal(integer,a['uint8_repeat']);assert repeat==case['repeat_exact']
   values=np.clip(np.pad(a['raw'],((0,(-len(a['raw']))%3),(0,0))).astype(np.float32)/25.6,0,26);sc,z=case['input_quantization'];q=np.clip(np.rint(values/sc)+z,-128,127).astype(np.int8);assert np.array_equal(q,a['quantized_input'])
   clocks=(3*np.arange(1,len(floating)+1)+2)*160;assert np.array_equal(clocks,a['support_unclamped']) and np.array_equal(np.minimum(clocks,row['samples']),a['sample_end'])
   os,oz=case['output_quantization'];signed=(integer.astype(np.float64)-oz)*os-floating;err=np.abs(np.convolve(signed,np.ones(5)/5,'valid'));stats=dict(mean=float(err.mean()),p99=float(np.quantile(err,.99)),maximum=float(err.max()));assert stats==case['smoothed_abs_error'];phase=float(np.max(np.abs(a['offline_phase']-floating[74:])));assert phase==case['phase_max_error']
   passed=bool(repeat and np.isfinite(floating).all() and phase<=g['float_offline_stream_max_abs'] and stats['mean']<=g['per_input_five_step_mean_abs_error_max'] and stats['p99']<=g['per_input_five_step_p99_abs_error_max'] and stats['maximum']<=g['per_input_five_step_max_abs_error_max']);assert passed==case['passed']
  name=case['condition']+'/'+case['target'];s=summaries.setdefault(name,dict(cases=0,passed=0,failed_requests=[],worst_mean=0,worst_p99=0,worst_maximum=0));s['cases']+=1;s['passed']+=passed
  if not passed:s['failed_requests'].append(case['request_order'])
  for k,v in stats.items():s['worst_'+k]=max(s['worst_'+k],v)
 assert all(s['cases']==135 for s in summaries.values()) and len(summaries)==4
 qualified=[condition for condition in ['twoway_consistency','oneway_consistency'] if all(summaries[condition+'/'+target]['passed']==135 for target in ['hey_willow','hey_nyx'])]
 assert result['all_passed']==all(c['passed'] for c in result['cases'])
 output=dict(result_sha256=sha(p),numerical_gates=g,summary=summaries,qualified_complete270_conditions=qualified,limits='Independent all540 savedSHA/repeats/inputquantization/completeclocks/phase/fivestats/gates. Numerical qualification only; no selection using freshdev/acceptance/recognition, no promotion or P4 claim.')
 with (R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-independent-audit-v2.json').open('x') as f:f.write(json.dumps(output,indent=2)+'\n')
 print(json.dumps(output,indent=2))
if __name__=='__main__':main()
