"""Reconstruct physical training support and recompute every saved crossing."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 meta=R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json';inventory=json.loads(meta.read_text())['inventory'];assert len(inventory)==4320
 arms={}
 for arm in ['control','decay']:
  report=R/f'4090-binary40-regularization-{arm}-physical-fit-original-v1.json';d=json.loads(report.read_text());plan=R/f'4090-binary40-regularization-{arm}-physical-fit-plan-v1.json'
  assert d['arm']==arm and d['optimizer_updates']==0 and d['plan_sha256']==sha(plan)
  for p,h in json.loads(plan.read_text())['files'].items():assert sha(p)==h,p
  trace=d['trace'];assert sha(trace['path'])==trace['sha256'] and Path(trace['path']).stat().st_size==trace['bytes'];assert len(d['records'])==4320
  totals={};identities=set()
  with np.load(trace['path']) as z:
   assert len(z.files)==8640
   for i,(row,source) in enumerate(zip(d['records'],inventory)):
    assert row['index']==i and row['source_id']==source['source']['id'] and row['label']==source['source']['label'];identities.add(row['source_id']);label=row['label']
    n=source['frames']//3;support=(3*np.arange(1,n+1)+2)*160;end=source['prefix_samples']+source['reference_samples']
    mask=((support>=end+480)&(support<=end+4320)) if label else support>source['prefix_samples']
    valid=np.convolve(mask.astype(int),np.ones(5,dtype=int),'valid')==5;assert valid.any()
    g=totals.setdefault(str(label),dict(rows=0,silence_crossings=0,speech_crossings=0));g['rows']+=1
    for view in ['silence','speech']:
     v=z[f'p_{i}_{view}'];assert v.shape==(n,) and np.isfinite(v).all() and np.all((v>=0)&(v<=1));five=np.convolve(v.astype(np.float64),np.ones(5)/5,'valid');maximum=float(five[valid].max());hit=bool(np.any(five[valid]>=.91796875));expected=row['arms'][view];assert hit==expected['crossing'] and abs(maximum-expected['max_five'])<=1e-12;g[view+'_crossings']+=hit
  assert totals==d['summary'] and len(identities)==4320
  arms[arm]=dict(report_sha256=sha(report),plan_sha256=sha(plan),trace_sha256=trace['sha256'],summary=totals,records=4320,traces=8640)
 out=dict(schema='binary40-regularization-physical-fit-independent-review-v1',all_passed=True,inventory_sha256=sha(meta),arms=arms,limits='Independent saved probability/mask/max/crossing recomputation, no rerun of network tensors. Training support only, not generalization, complete main-family fit, float/INT8 equivalence or P4. Both final students fit all864original physical positive sources in both histories; does not identify the cause of consumed ordinary regressions.')
 p=R/'4090-binary40-regularization-physical-fit-review-v1.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
