"""Full540 saved-trace diagnostic; original gate outcomes remain unchanged."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-bounded-consistency-fixed-grid-screen-result-v1.json';audit=R/'4090-hey-broad-bounded-consistency-fixed-grid-screen-independent-audit-v1.json';data=json.loads(p.read_text());assert json.loads(audit.read_text())['result_sha256']==sha(p);assert len(data['cases'])==540
 rows=[];summary={}
 for c in data['cases']:
  path=Path(c['arrays_path']);assert sha(path)==c['arrays_sha256']
  with np.load(path,allow_pickle=False) as a:
   f=a['float'].astype(np.float64);scale,zero=c['output_quantization'];q=(a['uint8'].astype(np.float64)-zero)*scale
   assert f.ndim==q.ndim==1 and len(f)==len(q)
   error=np.abs(np.convolve(q-f,np.ones(5)/5,'valid'));bad=np.flatnonzero(error>.1);starts=np.arange(len(error))
   # Window start>=74 means every element has full context225 support.
   post=starts>=74
   row=dict(target=c['target'],condition=c['condition'],request_order=c['request_order'],original_passed=c['passed'],arrays_sha256=c['arrays_sha256'],windows=len(error),over_point_one_windows=int(len(bad)),over_point_one_wholly_full_support=int(np.sum(bad>=74)),post_support_p99=float(np.quantile(error[post],.99)) if post.any() else None,post_support_maximum=float(error[post].max()) if post.any() else None,first_over_point_one_window_start=int(bad[0]) if len(bad) else None)
   rows.append(row);key=c['target']+'/'+c['condition'];s=summary.setdefault(key,dict(cases=0,original_failed=0,original_failed_with_full_support_over_point_one=0));s['cases']+=1;s['original_failed']+=int(not c['passed']);s['original_failed_with_full_support_over_point_one']+=int(not c['passed'] and row['over_point_one_wholly_full_support']>0)
 out=R/'4090-hey-broad-bounded-consistency-saved-window-diagnostic-v1.json'
 with out.open('x') as f:json.dump(dict(files={str(v):sha(v) for v in (Path(__file__).resolve().relative_to(Path.cwd()),p,audit)},summary=summary,cases=rows,model_invocations=0,fresh_material_reads=0,limits='All original540 saved traces, no model inference or filtering. Diagnostic .1 window cutoff is original p99-error limit used pointwise for location only; it does not replace original per-input gates. Index74 fullsupport225 from original offline/stream alignment. Post-support stats descriptive only; never delete startup or exempt original failures. Does not identify a causal layer, recognition effects or authorize fresh data.'),f,indent=2);f.write('\n')
 print(json.dumps(summary))
if __name__=='__main__':main()
