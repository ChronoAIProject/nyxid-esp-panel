"""Audit fixed-final training identities/Adam/entire original batch schedules."""
import hashlib,json
from collections import Counter
from pathlib import Path
import h5py,numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def datasets(path):
 values={}
 with h5py.File(path) as f:
  def visit(n,v):
   if isinstance(v,h5py.Dataset):
    a=v[()];assert np.isfinite(a).all();values[n]=(tuple(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest())
  f.visititems(visit)
 return values
def inv(path):return Counter((tuple(v['shape']),v['dtype'],v['sha256']) for v in json.loads(Path(path).read_text()))
def main():
 lp=R/'4090-pooled-matched-formal-execution-v1.json';p=json.loads(lp.read_text());rp=R/'4090-pooled-matched-formal-result-v1.json';r=json.loads(rp.read_text());assert r['all_passed'] and r['execution_sha256']==sha(lp) and len(r['cases'])==4
 original=json.loads(Path(p['original_formal']).read_text());rows=[]
 for c in r['cases']:
  target=c['target'];condition=c['condition'];folder=Path(p['output'])/f'{target}-{condition}-0';receipt=json.loads((folder/'training.json').read_text());assert receipt==c['final_repeat'] and receipt['steps']==10000 and receipt['parameter_delta_l2']>0
  source=next(a['initial'][condition] for a in p['arms'] if a['target']==target)
  initial=datasets(folder/'initial.weights.h5');final=datasets(folder/'final.weights.h5')
  assert Counter(initial.values())==Counter(datasets(source).values())==inv(folder/'initial-inventory.json')
  assert Counter(final.values())==inv(folder/'final-inventory.json')
  bn=[n for n in initial if 'batch_normalization' in n];assert len(bn)==16
  assert all(initial[n]==final[n] for n in bn)
  assert sha(folder/'final.weights.h5')==receipt['weights_sha256'] and sha(folder/'final-optimizer.npz')==receipt['optimizer_sha256']
  with np.load(folder/'final-optimizer.npz',allow_pickle=False) as a:
   vals=[a[f'v{i}'] for i in range(len(a.files))];assert len(vals)==32 and int(vals[0])==10000 and all(np.isfinite(x).all() for x in vals)
   assert hashlib.sha256(b''.join(v.tobytes() for v in vals)).hexdigest()==receipt['identity']['optimizer']
  actual=[json.loads(x) for x in (folder/'actual-batch-schedule.jsonl').read_text().splitlines()];wanted=[json.loads(x) for x in Path(original['arms'][target]['schedule']).read_text().splitlines()][:10000]
  assert len(actual)==len(wanted)==10000
  assert all(a['step']==i and a['batch_sha256']==w['tensor_sha256'] for i,(a,w) in enumerate(zip(actual,wanted,strict=True),1))
  rows.append(dict(target=target,condition=condition,steps=10000,initialization_exact=True,frozen_bn_arrays=16,adam_iterations=10000,weights_sha256=receipt['weights_sha256'],schedule_sha256=sha(folder/'actual-batch-schedule.jsonl')))
 with (R/'4090-pooled-matched-formal-independent-audit-v1.json').open('x') as f:f.write(json.dumps(dict(result_sha256=sha(rp),cases=rows,all_passed=True,limits='Saved checkpoints/Adam/original schedules independently audited; no model inference or recognition/INT8 qualification.'),indent=2)+'\n')
 print('All4 fixed-final10000 training artifacts audited')
if __name__=='__main__':main()
