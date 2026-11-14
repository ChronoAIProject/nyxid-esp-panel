"""Independent original smoke checkpoint/inventory/optimizer/schedule audit."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np,h5py
R=Path('research/wakeword_p4/results');p=R/'4090-pooled-matched-smoke-result-v1.json';r=json.loads(p.read_text());lock=json.loads((R/'4090-pooled-matched-smoke-execution-v1.json').read_text());original=json.loads(Path(lock['original_formal']).read_text());rows=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def numerical_h5(path):
 values=[]
 with h5py.File(path) as f:
  def collect(n,v):
   if isinstance(v,h5py.Dataset):
    a=v[()];values.append((tuple(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest()))
  f.visititems(collect)
 return Counter(values)
def inv(path):return Counter((tuple(v['shape']),v['dtype'],v['sha256']) for v in json.loads(Path(path).read_text()))
for case in r['cases']:
 runs=[];target=case['target'];condition=case['condition'];source=next(a['initial'][condition] for a in lock['arms'] if a['target']==target);expected=[json.loads(l) for l in Path(original['arms'][target]['schedule']).read_text().splitlines()][:2]
 for rep in range(2):
  out=Path(lock['output'])/f'{target}-{condition}-{rep}';receipt=json.loads((out/'training.json').read_text());assert receipt['steps']==2 and receipt['parameter_delta_l2']>0 and receipt['frozen_bn_exact']
  assert numerical_h5(out/'initial.weights.h5')==numerical_h5(source)==inv(out/'initial-inventory.json')
  assert numerical_h5(out/'final.weights.h5')==inv(out/'final-inventory.json');assert sha(out/'final.weights.h5')==receipt['weights_sha256'] and sha(out/'final-optimizer.npz')==receipt['optimizer_sha256']
  with np.load(out/'final-optimizer.npz') as a:
   vals=[a[f'v{n}'] for n in range(len(a.files))];assert int(vals[0])==2 and len(vals)==32 and all(np.isfinite(v).all() for v in vals)
   sig=hashlib.sha256(b''.join(v.tobytes() for v in vals)).hexdigest();assert sig==receipt['identity']['optimizer']
  schedule=[json.loads(l) for l in (out/'actual-batch-schedule.jsonl').read_text().splitlines()];assert len(schedule)==2
  for actual,wanted in zip(schedule,expected,strict=True):assert actual['batch_sha256']==wanted['tensor_sha256']
  runs.append(receipt['identity'])
 assert runs[0]==runs[1];rows.append(dict(target=target,condition=condition,actual_h5_numeric_inventory_exact=True,actual_optimizer_iterations=2,fresh_repeated_model_optimizer_identity_exact=True))
assert len(rows)==4 and r['all_passed']
out=R/'4090-pooled-matched-smoke-independent-audit-v1.json'
with out.open('x') as f:f.write(json.dumps(dict(result_sha256=sha(p),cases=rows,all_passed=True,limits='Checkpoint numerical inventories/optimizer/schedules independently audited; frozenBN live assertion preserved. Two updates only, no fullrun/candidate-quality claim.'),indent=2)+'\n')
print('All8 smoke checkpoints and Adam inventories independently verified')
