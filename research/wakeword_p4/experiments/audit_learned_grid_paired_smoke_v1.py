"""Independent saved post-update parity/state repeat audit; no model execution."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rp=R/'4090-learned-grid-paired-smoke-result-v1.json';lp=R/'4090-learned-grid-paired-smoke-execution-v1.json';r=json.loads(rp.read_text());p=json.loads(lp.read_text());assert r['execution_sha256']==sha(lp);rows=[];count=0
 for c in r['cases']:
  assert len(c['labels'])==len(c['weights'])==128 and len(c['repeats'])==2
  assert c['labels'].count(1.)==32 and c['labels'].count(0.)==96 and all(w==(1. if y else 20.) for y,w in zip(c['labels'],c['weights'],strict=True))
  reps=c['repeats'];assert reps[0]['model_sha256']==reps[1]['model_sha256'] and c['losses'][0]==c['losses'][1]
  for rep in reps:assert sha(rep['model_path'])==rep['model_sha256'] and sha(rep['arrays_path'])==rep['arrays_sha256'] and rep['compiler']['parameter_writes']
  assert reps[0]['history']==reps[1]['history'] and [h['step'] for h in reps[0]['history']]==[1,2]
  with np.load(reps[0]['arrays_path'],allow_pickle=False) as a,np.load(reps[1]['arrays_path'],allow_pickle=False) as b:
   assert set(a.files)==set(b.files)
   for k in a.files:assert np.isfinite(a[k]).all() and a[k].tobytes()==b[k].tobytes()
   names=[k for k in a.files if k.isdigit()]+['embedding','logit'];assert len(names)==11
   for k in names:
    x=a['tf_'+k];assert np.max(abs(x-np.rint(x)))<1e-10;np.testing.assert_array_equal(np.rint(x),a[k].reshape(x.shape));count+=x.size*2
   variables=[k for k in a.files if k.startswith('variable_')];optimizer=[k for k in a.files if k.startswith('optimizer_')]
   assert len(variables)==21 # nine logsteps are one vector +20 folded W/b tensors
   assert a['variable_0'].shape==(9,)
   assert (np.any(a['variable_0']!=0) if c['training_arm']=='learned_step' else np.all(a['variable_0']==0))
   assert int(a['optimizer_0'])==2 and len(optimizer)==(44 if c['training_arm']=='learned_step' else 42) #21 variables,2 slots each +iteration/lr
   rows.append(dict(target=c['target'],condition=c['condition'],training_arm=c['training_arm'],parameter_tensors=20,activation_step_values=9,optimizer_arrays=len(optimizer),passed=True))
 assert len(rows)==4
 for word in ('hey_willow','hey_nyx'):
  arms=[c for c in r['cases'] if c['target']==word];assert len(arms)==2
  assert [h['batch_sha256'] for h in arms[0]['repeats'][0]['history']]==[h['batch_sha256'] for h in arms[1]['repeats'][0]['history']]
 out=dict(result_sha256=sha(rp),auditor_sha256=sha(__file__),cases=rows,all_passed=True,code_values_both_repeats=count,model_invocations=0,limits='Saved postupdate TF/actual codes plus parameter/Adam/export-repeat identity. Two updates on first2 full128 originalbatches only, paired fixedgrid/learnedstep; no full540 numerical qualification, fresh recognition, FAR or training efficacy evidence.')
 (R/'4090-learned-grid-paired-smoke-saved-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
