"""Independent H5/Adam/history audit of fixed-final encoder-freezing runs."""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/frozen-encoder-formal-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 out={}
 with h5py.File(p) as f:
  def visit(k,v):
   if isinstance(v,h5py.Dataset):out[k]=v[()]
  f.visititems(visit)
 return out
def identical(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 path=R/'4090-frozen-encoder-formal-result-v1.json'
 if not path.exists():print(json.dumps({'status':'not_ready','missing':str(path)}));raise SystemExit(2)
 r=json.loads(path.read_text());lockpath=R/'4090-frozen-encoder-formal-execution-v1.json';d=json.loads(lockpath.read_text());assert sha(lockpath)==r['execution_sha256'] and len(r['cases'])==4 and r['all_passed']
 init=arrays(d['initial']['path']);assert sha(d['initial']['path'])==d['initial']['sha256'];heads={k for k in init if k.startswith('layers/dense/')};assert len(heads)==2
 checks=[]
 for case in r['cases']:
  target,condition=case['target'],case['condition'];dest=OUT/f'{target}-{condition}';original=arrays(dest/'initial.weights.h5');assert original.keys()==init.keys() and all(identical(original[k],init[k]) for k in init)
  for key in ['weights','optimizer','history']:assert sha(case[key+'_path'])==case[key+'_sha256']
  schedule=[json.loads(l) for l in Path(d['schedules'][target]['path']).read_text().splitlines()];history=[json.loads(l) for l in Path(case['history_path']).read_text().splitlines()];assert len(history)==len(schedule)==10000
  for step,(row,want) in enumerate(zip(history,schedule,strict=True),1):
   assert row['step']==step and row['tensor_sha256']==want['phase_sha256'] and row['original_batch_sha256']==want['original_receipt']['tensor_sha256'] and np.isfinite(row['loss'])
  frozen_keys=set(init)-heads if condition=='frozen' else {k for k in init if 'batch_normalization' in k}
  for step in range(1000,10001,1000):
   a=arrays(dest/f'step{step}.weights.h5');assert a.keys()==init.keys()
   for k,v in a.items():
    assert v.shape==init[k].shape and v.dtype==init[k].dtype and np.isfinite(v).all()
    if k in frozen_keys:assert identical(v,init[k])
   with np.load(dest/f'step{step}-optimizer.npz',allow_pickle=False) as opt:
    count=2+2*case['trainables'];assert opt.files==[f'v{i}' for i in range(count)]
    assert int(opt['v0'])==step and opt['v1'].shape==() and opt['v1']==np.float32(1e-5)
    assert all(np.isfinite(opt[k]).all() for k in opt.files)
  final=arrays(case['weights_path']);snapshot=arrays(dest/'step10000.weights.h5');assert final.keys()==snapshot.keys() and all(identical(final[k],snapshot[k]) for k in final)
  assert all(not identical(final[k],init[k]) for k in heads)
  with np.load(case['optimizer_path'],allow_pickle=False) as finalopt,np.load(dest/'step10000-optimizer.npz',allow_pickle=False) as snapopt:
   assert finalopt.files==snapopt.files and all(identical(finalopt[k],snapopt[k]) for k in finalopt.files)
   assert hashlib.sha256(b''.join(finalopt[k].tobytes() for k in finalopt.files)).hexdigest()==case['optimizer_signature']
  checks.append({'target':target,'condition':condition,'steps':10000,'snapshots':10,'all_input_hashes_match':True,'original_init_exact':True,'frozen_h5_arrays_exact':True,'final_equals_step10000':True,'adam_arrays_finite_signature_and_iteration_exact':True})
 for target in d['targets']:
  histories=[[json.loads(l) for l in (OUT/f'{target}-{c}'/'history.jsonl').read_text().splitlines()] for c in d['conditions']]
  assert all(a['tensor_sha256']==b['tensor_sha256'] and a['phase_input_sha256']==b['phase_input_sha256'] for a,b in zip(*histories,strict=True))
 result={'source_result_sha256':sha(path),'checks':checks,'all_passed':True,'limits':'Independent saved H5/Adam/history checks; model runtime variable-order signature and connected-gradient assertions remain runtime evidence, not reconstructed from H5 path ordering. No recognition qualification.'}
 with (R/'4090-frozen-encoder-formal-independent-audit-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
