"""Independent precision-control array comparisons; never invokes TensorFlow."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compare(reference,observed,atol,rtol):
 assert reference.keys()==observed.keys();details={}
 for k in reference:
  a=reference[k].astype(float);b=observed[k].astype(float);assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all();delta=abs(a-b);limit=atol+rtol*abs(a);details[k]=dict(shape=list(a.shape),values=a.size,max_abs=float(delta.max()),violating_values=int(np.count_nonzero(delta>limit)),max_tolerance_ratio=float(np.max(delta/limit)),passed=bool(np.all(delta<=limit)))
 return dict(all_arrays_passed=all(x['passed'] for x in details.values()),arrays=details)
def main():
 lock=R/'4090-engine-gradient-precision-execution-v1.json';result=R/'4090-engine-gradient-precision-result-v1.json';r=json.loads(result.read_text());d=json.loads(lock.read_text());assert r['all_workers_completed'] and r['execution_sha256']==sha(lock) and r['optimizer_updates']==0 and len(r['reports'])==4;allcases={};settings={};baseline=[]
 for report in r['reports']:
  assert sha(report['path'])==report['sha256'];mode=report['mode'];w=json.loads(Path(report['path']).read_text());s=w['settings'];settings[mode]=s;assert s['mode']==mode and s['eager'] and s['optimizer_created'] is False and s['intraop']==2
  if mode.endswith('off'):assert s['tf32_after'] is False
  else:assert s['tf32_before']==s['tf32_after']
  assert len(s['visible_gpu_names'])==(1 if mode.startswith('gpu') else 0)
  assert w['all_mechanics_passed'] and len(w['cases'])==2
  for case in w['cases']:
   assert case['repeat_exact'] and case['state_unchanged'] and case['snapshot_all33_arrays_exact'] and sha(case['arrays_path'])==case['arrays_sha256'] and sha(case['original_arrays_path'])==case['original_arrays_sha256']
   assert ('/device:GPU:0' if mode.startswith('gpu') else '/device:CPU:0') in case['actual_logits_device']
   with np.load(case['arrays_path'],allow_pickle=False) as a:values={k:a[k].copy() for k in a.files}
   with np.load(case['original_arrays_path'],allow_pickle=False) as a:original={k:a[k].copy() for k in a.files if k.startswith(('S_','D_')) or k in ['main_logits','domain_logits']};original['losses']=a['losses'][:2].copy()
   assert all(k in values for k in original);exact=all(np.array_equal(values[k],v) for k,v in original.items());assert exact==case['original_gpu_common_arrays_exact'];assert all(np.isfinite(v).all() for v in values.values());allcases[mode,case['target']]=values
   if mode=='gpu-default':baseline.append(dict(target=case['target'],all_original_common_arrays_exact=exact))
 assert len(allcases)==8 and len(baseline)==2
 pairs=[('gpu-default','cpu-default'),('cpu-default','gpu-off'),('gpu-default','gpu-off'),('cpu-default','cpu-off')];comparisons=[]
 for target in ['hey_nyx','hey_willow']:
  for reference,observed in pairs:comparisons.append(dict(target=target,reference_mode=reference,observed_mode=observed,**compare(allcases[reference,target],allcases[observed,target],**d['pointwise_rule'])))
 out=dict(source_result_sha256=sha(result),execution_sha256=sha(lock),auditor_sha256=sha(__file__),all_mechanics_and_array_hash_checks_passed=True,original_gpu_default_reproduction=baseline,valid_original_baseline_isolation=all(x['all_original_common_arrays_exact'] for x in baseline),pointwise_rule=d['pointwise_rule'],settings=settings,comparisons=comparisons,formal_process_before=r['formal_before'],formal_process_after=r['formal_after'],optimizer_updates=0,limits='Only original batch1 per word on same Linux host, four fresh processes. All numerical failures retained; per-array comparison is not a whole-network/autodiff oracle or evidence of recognition causality/INT8/freshdata qualification. New embeddings have no original-saved GPU counterpart. Live formal was not modified.')
 with (R/'4090-engine-gradient-precision-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print('Original default GPU replay exact:',out['valid_original_baseline_isolation'])
 for p in comparisons:print(json.dumps(dict(target=p['target'],reference=p['reference_mode'],observed=p['observed_mode'],all_arrays_passed=p['all_arrays_passed'],worst_abs=max(x['max_abs'] for x in p['arrays'].values()),violations=sum(x['violating_values'] for x in p['arrays'].values()))))
 return 0 if out['valid_original_baseline_isolation'] else 1
if __name__=='__main__':raise SystemExit(main())
