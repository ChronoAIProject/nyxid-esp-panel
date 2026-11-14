"""Independent saved TF/actual nonuniform code/repeat audit; no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rp=R/'4090-learned-grid-nonuniform-stream-result-v1.json';lp=R/'4090-learned-grid-nonuniform-stream-execution-v1.json';r=json.loads(rp.read_text());p=json.loads(lp.read_text());assert r['execution_sha256']==sha(lp);rows=[];count=0
 for c in r['cases']:
  assert sha(c['compiled_path'])==c['compiled_sha256'] and sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a:
   keys=sorted([k for k in a.files if k.startswith('actual_') and k[7:].isdigit()],key=lambda k:int(k[7:]));assert len(keys)==9
   for k in a.files:assert np.isfinite(a[k]).all() and a[k][0].tobytes()==a[k][1].tobytes()
   for j,k in enumerate(keys):
    x=a[f'layer_{j}'];expected=a[k].reshape(x.shape);assert np.max(abs(x-np.rint(x)))<1e-10;np.testing.assert_array_equal(np.rint(x),expected);count+=x.size
   g=a['step_gradients'];assert g.shape==(2,9) and np.all(g!=0) and g[0].tolist()==c['step_gradients']
   rows.append(dict(target=c['target'],condition=c['condition'],passed=True))
 assert len(rows)==4
 out=dict(result_sha256=sha(rp),auditor_sha256=sha(__file__),fixed_log_ratios=p['ratios'],cases=rows,all_passed=True,encoder_values_both_repeats=count,model_invocations=0,limits='Saved actual exported INT8 vs TF encoder codes/repeats and diagnostic gradients only. Does not independently verify compiler constraints/arithmetic, trained folded weights/bias or FC and recognition.')
 (R/'4090-learned-grid-nonuniform-saved-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
