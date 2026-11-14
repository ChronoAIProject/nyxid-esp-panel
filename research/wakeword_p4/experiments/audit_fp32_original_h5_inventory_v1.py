"""Read original H5 datasets and match every saved runtime parameter value identity."""
from collections import Counter
import hashlib,json
from pathlib import Path
import h5py
R=Path('research/wakeword_p4/results')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 lp=R/'4090-fp32-internal-software-execution-v1.json';rp=R/'4090-fp32-internal-software-original-result-v1.json';lock=json.loads(lp.read_text());result=json.loads(rp.read_text());cases=[]
 for m,r in zip(lock['models'],result['cases'],strict=True):
  assert (m['target'],m['condition'])==(r['target'],r['condition']) and sha(Path(m['path']).read_bytes())==m['sha256']==r['h5_sha256'];datasets=[]
  with h5py.File(m['path'],'r') as f:
   def visit(n,o):
    if isinstance(o,h5py.Dataset):
     a=o[()];datasets.append(dict(path=n,shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes())))
   f.visititems(visit)
  def key(x):return tuple(x['shape']),x['dtype'],x['sha256']
  assert Counter(map(key,datasets))==Counter(map(key,r['inventory']))
  bn=[x for x in datasets if x['path'].startswith('layers/batch_normalization')];assert len(bn)==16
  cases.append(dict(target=m['target'],condition=m['condition'],h5_sha256=m['sha256'],datasets=datasets,all_dataset_multiset_exact=True,bn_dataset_count=len(bn),runtime_variable_count=len(r['inventory'])))
 out=dict(schema='fp32-original-h5-inventory-audit-v1',files={str(p):sha(p.read_bytes()) for p in [lp,rp,Path(__file__)]},cases=cases,limits='Read-only original H5 datasets and saved runtime inventory. Exact shape/dtype/value-hash multisets with multiplicity, not a positional variable-path mapping or independent original forward execution. No TensorFlow import, inference/export/training/media access. BN values included; original unchanged-after-run assertion remains producer evidence.')
 (R/'4090-fp32-original-h5-inventory-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([dict(target=c['target'],condition=c['condition'],datasets=len(c['datasets']),bn=c['bn_dataset_count']) for c in cases]))
if __name__=='__main__':main()
