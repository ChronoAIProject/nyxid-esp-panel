"""Independent NPZ reduction of failed gradient-additivity diagnostic."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-consistency-gradient-diagnostic-result-v1.json';r=json.loads(p.read_text());summary=[]
 assert len(r['cases'])==8
 for c in r['cases']:
  assert sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a:
   h=hashlib.sha256()
   for k in ['x','y','w']:h.update(a[k].tobytes())
   assert h.hexdigest()==c['batch_sha256'] and np.array_equal(a['losses'],a['repeat_losses'])
   checks=[];sumerror2=0;total2=0
   for j,reported in enumerate(c['component_comparison']):
    s,g,t=[a[f'{name}_{j}'] for name in ['supervised','consistency','total']]
    for name in ['supervised','consistency','total']:assert np.isfinite(a[f'{name}_{j}']).all() and np.array_equal(a[f'{name}_{j}'],a[f'repeat_{name}_{j}'])
    error=np.abs((s+g).astype(np.float64)-t);outside=int(np.sum(error>1e-5+1e-5*np.abs(t.astype(np.float64))))
    assert outside==reported['outside_original_tolerance'] and float(error.max())==reported['maximum_abs']
    sumerror2+=float(np.sum(error**2));total2+=float(np.sum(t.astype(np.float64)**2));checks.append(outside)
   row=dict(target=c['target'],batch=c['batch'],outside_original_tolerance=sum(checks),relative_l2_difference=float(np.sqrt(sumerror2/total2)),original_gate_passed=all(n==0 for n in checks),repeat_exact=True)
   assert row['original_gate_passed']==c['original_additivity_tolerance_passed'];summary.append(row)
 assert r['original_additivity_all_passed']==all(c['original_gate_passed'] for c in summary)
 out=R/'4090-hey-broad-consistency-gradient-diagnostic-independent-audit-v1.json'
 with out.open('x') as f:json.dump(dict(result_sha256=sha(p),summary=summary,limits='Saved gradients authenticate repeated arrays and recompute original elementwise1e-5 criterion plus descriptive relativeL2. RelativeL2 does not replace failed original criterion. No independent network autodiff, model updates or admission.'),f,indent=2);f.write('\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
