"""Independent saved-gradient and actual fresh-smoke artifact reduction."""
import hashlib,json,math
from pathlib import Path
import h5py
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrays(path):
 out={}
 with h5py.File(path) as f:
  def visit(name,value):
   if isinstance(value,h5py.Dataset):out[name]=value[()]
  f.visititems(visit)
 return out

def main():
 path=R/'4090-phase-aligned-preflight-result-v1.json';x=json.loads(path.read_text());assert x['all_passed'] and x['gradient_variables_unchanged'] and len(x['gradient_probe'])==50
 assert x['gpu']['device_name']=='NVIDIA GeForce RTX 4090'
 for index,row in enumerate(x['gradient_probe'],1):
  assert row['step']==index and sha(row['gradient_path'])==row['gradient_sha256']
  with np.load(row['gradient_path'],allow_pickle=False) as a:
   assert set(a.files)=={'nominal','phase'}
   for c in row['conditions']:
    v=a[c['condition']];assert v.ndim==1 and v.dtype==np.float32 and np.isfinite(v).all()
    assert hashlib.sha256(v.tobytes()).hexdigest()==c['gradient_sha256']
    norm=math.sqrt(math.fsum(float(z)**2 for z in v));assert math.isclose(norm,c['gradient_l2'],rel_tol=1e-12,abs_tol=1e-12) and math.isfinite(c['loss'])
   n=a['nominal'].astype(float);p=a['phase'].astype(float);cos=float(np.dot(n,p)/(np.linalg.norm(n)*np.linalg.norm(p)));assert math.isclose(cos,row['gradient_cosine'],rel_tol=1e-12,abs_tol=1e-12)
 for arm in x['smoke']:
  assert arm['repeat_exact'] and len(arm['repeats'])==2
  a,b=arm['repeats']
  for r in (a,b):
   assert r['initial_signature']==x['initial_signature'] and r['optimizer_iterations']==2 and r['frozen_bn_exact']
   assert sha(r['weights_path'])==r['weights_sha256'] and sha(r['optimizer_path'])==r['optimizer_sha256']
  left,right=arrays(a['weights_path']),arrays(b['weights_path']);assert left.keys()==right.keys() and left
  assert all(left[k].dtype==right[k].dtype and left[k].shape==right[k].shape and left[k].tobytes()==right[k].tobytes() and np.isfinite(left[k]).all() for k in left)
  with np.load(a['optimizer_path'],allow_pickle=False) as l,np.load(b['optimizer_path'],allow_pickle=False) as r:
   assert l.files==r.files and len(l.files)==32
   assert all(l[k].dtype==r[k].dtype and l[k].shape==r[k].shape and l[k].tobytes()==r[k].tobytes() and np.isfinite(l[k]).all() for k in l.files)
  assert a['history']==b['history'] and len(a['history'])==2
 result=dict(original_result_sha256=sha(path),gradient_batches=50,gradient_arrays=100,smoke_conditions=2,smoke_fresh_repeats=2,smoke_updates_per_repeat=2,saved_weights_and_adam_arrays_exact=True,all_passed=True,limits='Saved-artifact independent arithmetic/identity audit; not a second model execution or independent source-generalization result.')
 with (R/'4090-phase-aligned-preflight-independent-audit-v1.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
