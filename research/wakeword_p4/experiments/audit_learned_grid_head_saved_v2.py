"""Independent saved-array original-step integration audit, no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rp=R/'4090-learned-grid-head-integration-result-v2.json';lp=R/'4090-learned-grid-head-integration-execution-v2.json';r=json.loads(rp.read_text());caps=json.loads((R/'4090-matched-internal-boundaries-original-result-v1.json').read_text());lock=json.loads(lp.read_text());assert r['execution_sha256']==sha(lp);rows=[];values=0
 for c in r['cases']:
  prior=next(x for x in lock['cases'] if (x['target'],x['condition'])==(c['target'],c['condition']))
  assert sha(c['arrays_path'])==c['arrays_sha256'] and sha(prior['arrays_path'])==prior['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a,np.load(prior['arrays_path'],allow_pickle=False) as b:
   keys=sorted([k for k in b.files if k.isdigit()],key=int);assert len(keys)==9
   fractional=0.
   for j,k in enumerate(keys):
    x=a[f'layer_{j}'];expected=b[k].reshape(x.shape)
    assert x[0].tobytes()==x[1].tobytes() and np.isfinite(x).all()
    fractional=max(fractional,float(np.max(abs(x-np.rint(x)))))
    np.testing.assert_array_equal(np.rint(x),expected);values+=x.size
   g=a['step_gradients'];assert g.shape==(2,9) and np.isfinite(g).all() and np.all(g!=0) and g[0].tobytes()==g[1].tobytes();assert g[0].tolist()==c['step_gradients'];assert fractional<1e-10
   for j,norm in enumerate(c['parameter_gradient_norms']):
    pg=a[f'parameter_gradient_{j}'];assert np.isfinite(pg).all() and pg[0].tobytes()==pg[1].tobytes();assert float(np.linalg.norm(pg[0]))==norm
   assert len(c['parameter_gradient_norms'])==20
   cap=next(x for x in caps['cases'] if (x['target'],x['condition'])==(c['target'],c['condition']))
   assert sha(cap['captures_path'])==cap['captures_sha256']
   with np.load(cap['captures_path'],allow_pickle=False) as old:
    for name,key in [('embedding','int8_embedding'),('logit_codes','int8_logit')]:
     x=a[name];assert x[0].tobytes()==x[1].tobytes();assert np.max(abs(x-np.rint(x)))<1e-10;np.testing.assert_array_equal(np.rint(x),old[key].reshape(x.shape))
   rows.append(dict(target=c['target'],condition=c['condition'],max_fractional_code_residue=fractional,passed=True))
 assert len(rows)==4
 out=dict(result_sha256=sha(rp),auditor_sha256=sha(__file__),cases=rows,encoder_values_both_repeats=values,all_passed=True,model_invocations=0,limits='Saved full original-step encoder codes/repeat and embedding/logit codes and nine diagnostic step gradients plus20 parameter gradients; zero headbias gradient allowed for symmetric linear diagnostic objective. Does not prove changed-step TF/export or full trainable-weight/bias/head/recognition efficacy.')
 (R/'4090-learned-grid-head-saved-audit-v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
