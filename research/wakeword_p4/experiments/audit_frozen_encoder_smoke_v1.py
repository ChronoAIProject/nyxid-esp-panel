"""Saved H5 byte audit, independent of TF runtime variable signatures."""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(p):
 out={}
 with h5py.File(p) as f:
  def visit(k,v):
   if isinstance(v,h5py.Dataset):out[k]=v[()]
  f.visititems(visit)
 return out
def main():
 path=R/'4090-frozen-encoder-smoke-result-v1.json';r=json.loads(path.read_text());lockpath=R/'4090-frozen-encoder-smoke-execution-v1.json';lock=json.loads(lockpath.read_text());assert r['execution_sha256']==sha(lockpath)
 original=arrays(lock['initial']['path']);assert sha(lock['initial']['path'])==lock['initial']['sha256']
 heads={k for k in original if k.startswith('layers/dense/')};assert len(heads)==2
 checks=[]
 for case in r['cases']:
  saved=[]
  for repeat in case['repeats']:
   p=repeat['weights_path'];assert sha(p)==repeat['weights_sha256'];a=arrays(p);assert a.keys()==original.keys()
   for k,v in a.items():assert v.dtype==original[k].dtype and v.shape==original[k].shape and np.isfinite(v).all()
   changed=[k for k in a if a[k].tobytes()!=original[k].tobytes()]
   assert heads.issubset(changed)
   assert all('batch_normalization' not in k for k in changed)
   if case['condition']=='frozen':assert set(changed)==heads
   else:assert set(changed)-heads
   saved.append(a)
  assert all(saved[0][k].tobytes()==saved[1][k].tobytes() for k in original)
  checks.append({'target':case['target'],'condition':case['condition'],'changed_datasets':changed,'repeat_h5_arrays_exact':True})
 result={'source_result_sha256':sha(path),'checks':checks,'all_passed':True,'limits':'Independent stored H5 array audit only. Runtime optimizer/gradient/input signatures are not independently replayed. No recognition or INT8 qualification.'}
 with (R/'4090-frozen-encoder-smoke-independent-audit-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
