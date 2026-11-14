"""Independent original/final H5, BN, Adam and all actual training SHA audit."""
import hashlib,json
from pathlib import Path
import numpy as np
from audit_common_carrier_raw_training_v1 import datasets
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 resultpath=R/'4090-phase-aligned-formal-result-v1.json'
 if not resultpath.exists():print(json.dumps(dict(status='not_ready',missing=str(resultpath))));return 2
 lockpath=R/'4090-phase-aligned-formal-execution-v1.json';p=json.loads(lockpath.read_text());result=json.loads(resultpath.read_text());assert result['all_passed'] and result['execution_sha256']==sha(lockpath)
 for f,h in p['files'].items():assert sha(f)==h,f
 assert len(result['cases'])==2 and {c['condition'] for c in result['cases']}=={'nominal','phase'}
 schedule=[json.loads(l) for l in Path(p['schedule']).read_text().splitlines()];assert len(schedule)==10000
 original=datasets(p['initial']['initial_weights_path']);reports=[]
 for c in result['cases']:
  out=Path('.local/wakeword/phase-aligned-formal-v1')/c['condition'];assert json.loads((out/'training.json').read_text())==c and c['steps']==10000 and c['frozen_bn_exact']
  first=datasets(out/'initial.weights.h5');final=datasets(c['weights_path']);assert original.keys()==first.keys()==final.keys()
  assert all(first[n].dtype==original[n].dtype and first[n].shape==original[n].shape and first[n].tobytes()==original[n].tobytes() for n in original)
  assert any(first[n].tobytes()!=final[n].tobytes() for n in first) and all(np.isfinite(v).all() for v in final.values())
  bn=[n for n in first if n.startswith('layers/batch_normalization')];assert len(bn)==16 and all(first[n].tobytes()==final[n].tobytes() for n in bn)
  assert sha(c['weights_path'])==c['weights_sha256'] and sha(c['optimizer_path'])==c['optimizer_sha256'] and sha(c['history_path'])==c['history_sha256']
  with np.load(c['optimizer_path'],allow_pickle=False) as a:
   assert len(a.files)==32;vs=[a[f'v{i}'] for i in range(32)];assert int(vs[0])==10000 and np.array_equal(vs[1],np.float32(p['optimizer']['learning_rate'])) and all(np.isfinite(v).all() for v in vs)
   assert hashlib.sha256(b''.join(v.tobytes() for v in vs)).hexdigest()==c['optimizer_signature']
  history=[json.loads(l) for l in Path(c['history_path']).read_text().splitlines()];assert len(history)==10000
  for i,(a,b) in enumerate(zip(history,schedule,strict=True),1):assert a['step']==b['step']==i and a['tensor_sha256']==b[c['condition']+'_sha256'] and np.isfinite(a['loss'])
  for i in range(1000,10001,1000):
   snapshot=datasets(out/f'step{i}.weights.h5');assert snapshot.keys()==final.keys() and all(np.isfinite(v).all() for v in snapshot.values()) and all(first[n].tobytes()==snapshot[n].tobytes() for n in bn)
   assert (out/f'step{i}.index').exists() and list(out.glob(f'step{i}.data-*'))
  assert all(final[n].tobytes()==snapshot[n].tobytes() for n in final)
  reports.append(dict(condition=c['condition'],steps=10000,initial_all_arrays_exact=True,bn16_arrays_exact=True,adam32_arrays_signature_and_iteration_exact=True,actual_batch_hashes=10000,final_equals_step10000=True))
 report=dict(original_result_sha256=sha(resultpath),source_sha256=sha(__file__),cases=reports,all_passed=True,limits='Stored H5 datasets, final snapshot, Adam array-order signature and complete input histories audited without TF/model calls. Model variable-order signature remains runtime evidence, not independently reconstituted from H5 ordering. Not recognition/FAR/INT8 qualification.')
 with (R/'4090-phase-aligned-formal-independent-audit-v1.json').open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report));return 0
if __name__=='__main__':raise SystemExit(main())
