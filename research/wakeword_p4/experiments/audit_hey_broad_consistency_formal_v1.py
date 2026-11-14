"""Postcompletion formal schedule/checkpoint/Adam/BN audit; no inference."""
import hashlib,json
from pathlib import Path
from collections import Counter
import h5py,numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def values(path):
 out={}
 with h5py.File(path) as f:
  def visit(n,d):
   if isinstance(d,h5py.Dataset):out[n]=d[()]
  f.visititems(visit)
 return out
def identities(v):return Counter((tuple(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest()) for a in v.values())
def main():
 p=R/'4090-hey-broad-consistency-formal-result-v1.json';result=json.loads(p.read_text());lock=json.loads((R/'4090-hey-broad-consistency-formal-execution-v1.json').read_text());original=json.loads(Path(lock['original_formal']).read_text());reports=[]
 for case in result['cases']:
  c=case['final_repeat'];assert case['steps']==c['steps']==10000 and c['frozen_bn_exact'] and c['parameter_delta_l2']>0;out=Path(c['weights_path']).parent;arm=next(a for a in lock['arms'] if a['target']==case['target']);first=values(out/'initial.weights.h5');last=values(out/'final.weights.h5');assert identities(first)==identities(values(arm['path']))
  for label,data in [('initial',first),('final',last)]:
   inv=json.loads((out/f'{label}-inventory.json').read_text());assert identities(data)==Counter((tuple(a['shape']),a['dtype'],a['sha256']) for a in inv)
  bn=[n for n in first if 'batch_normalization' in n];assert len(bn)==16;assert all(np.array_equal(first[n],last[n]) for n in bn)
  assert sha(out/'final.weights.h5')==c['weights_sha256'] and sha(out/'final-optimizer.npz')==c['optimizer_sha256']
  with np.load(out/'final-optimizer.npz') as a:
   vals=[a[f'v{i}'] for i in range(len(a.files))];assert len(vals)==32 and int(vals[0])==10000 and all(np.isfinite(v).all() for v in vals);assert hashlib.sha256(b''.join(v.tobytes() for v in vals)).hexdigest()==c['identity']['optimizer']
  wanted=[json.loads(l) for l in Path(original['arms'][case['target']]['schedule']).read_text().splitlines()][:10000];actual=[json.loads(l) for l in (out/'actual-batch-schedule.jsonl').read_text().splitlines()];assert len(actual)==10000
  for index,(a,b) in enumerate(zip(actual,wanted,strict=True),1):assert a['step']==index and a['batch_sha256']==b['tensor_sha256'] and np.isfinite(a['loss']) and np.isfinite(a['gradient_l2']) and all(np.isfinite(a[k]) for k in ['supervised_loss','consistency_loss','supervised_gradient_l2','consistency_gradient_l2'])
  for step in range(1000,10001,1000):assert (out/f'step{step}.weights.h5').exists() and (out/f'step{step}.index').exists()
  reports.append(dict(target=case['target'],condition=case['condition'],steps=10000,original_initial_values_exact=True,frozen_bn_16arrays_exact=True,all_batch_schedule_hashes_exact=True,adam_iteration10000=True,all_periodic_checkpoints_present=True,weights_sha256=c['weights_sha256']))
 assert len(reports)==4 and result['all_passed']
 with (R/'4090-hey-broad-consistency-formal-independent-audit-v1.json').open('x') as f:f.write(json.dumps(dict(result_sha256=sha(p),cases=reports,limits='Training provenance/finite/checkpoint/BN audit only, no model numerical/recognition admission.'),indent=2)+'\n')
 print('All four fixed10000 formal runs independently audited')
if __name__=='__main__':main()
