"""Replay first original training batch only; no model or optimizer."""
import hashlib,json
from pathlib import Path
import numpy as np
from schedule_hey_broad_v1 import sampler
from hey_batch_schedule_v1 import batch_digest
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/learned-grid-smoke-batches-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parent=R/'4090-pooled-matched-formal-execution-v1.json';p=json.loads(parent.read_text());formal=Path(p['original_formal']);f=json.loads(formal.read_text())
 # Validate original sampler/data identities before reconstructing a consumed batch.
 for file,h in p['files'].items():assert sha(file)==h,file
 OUT.mkdir(exist_ok=False);rows=[]
 for target in ('hey_willow','hey_nyx'):
  schedule=Path(f['arms'][target]['schedule']);expected=json.loads(schedule.open().readline())
  env=sampler(target);x,y,w=env['batch'](mine=False,augment=False);digest=batch_digest(x,y,w);assert digest==expected['tensor_sha256']
  assert x.shape==(128,225,40) and y.shape==(128,1) and w.shape==(128,) and np.isfinite(x).all()
  assert np.all(np.isin(y,[0,1])) and np.all(w[y.reshape(-1)==1]==1) and np.all(w[y.reshape(-1)==0]==20)
  positive=int(np.flatnonzero(y.reshape(-1)==1)[0]);negative=int(np.flatnonzero(y.reshape(-1)==0)[0])
  dest=OUT/(target+'.npz');np.savez_compressed(dest,x=x,y=y,weight=w)
  rows.append(dict(target=target,batch_index=1,batch_sha256=digest,schedule_path=str(schedule),schedule_sha256=sha(schedule),arrays_path=str(dest),arrays_sha256=sha(dest),x_range=[float(x.min()),float(x.max())],labels_count={str(v):int(np.count_nonzero(y==v)) for v in (0,1)},prospective_smoke_rows=[positive,negative],policy='First positive and first negative by original batch index, not score selection; full128 original batch retained. Disposable integration smoke only; not representative recognition or formal training budget.'))
 out=dict(parent_sha256=sha(parent),original_formal_sha256=sha(formal),script_sha256=sha(__file__),cases=rows,model_invocations=0,optimizer_updates=0,limits='Only original consumed training sampler replay; all labels and negative20 weights authentic original. No fresh media/development/acceptance. Numerical phase/endpoint semantics for future streaming loss must be explicit before updates.')
 (R/'4090-learned-grid-smoke-original-batches-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
