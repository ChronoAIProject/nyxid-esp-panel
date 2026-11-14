"""Independent NumPy saved-array reduction; no TensorFlow, model or feature access."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
RESULT=R/'4090-frozen-encoder-training-witness-result-v1.json'
OUT=R/'4090-frozen-encoder-training-witness-independent-audit-v1.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 d=json.loads(RESULT.read_text());lock=R/'4090-frozen-encoder-training-witness-execution-v1.json';plan=json.loads(lock.read_text());assert d['all_passed'] and d['execution_sha256']==sha(lock)
 assert len(d['cases'])==4 and {(c['target'],c['condition']) for c in d['cases']}=={(t,c) for t in ['hey_nyx','hey_willow'] for c in ['full','frozen']}
 summaries=[];matched={}
 for c in d['cases']:
  assert sha(c['arrays_path'])==c['arrays_sha256'] and sha(c['receipts_path'])==c['receipts_sha256'] and c['rows']==12800 and c['repeat_exact'] and c['model_unchanged']
  source=next(a for a in plan['cases'] if a['target']==c['target'] and a['condition']==c['condition']);assert c['weights_sha256']==source['weights_sha256']
  schedule=plan['schedules'][c['target']];assert sha(schedule['path'])==schedule['sha256']
  original=[json.loads(l) for l in Path(schedule['path']).read_text().splitlines()][:100];receipts=json.loads(Path(c['receipts_path']).read_text());assert len(receipts)==100
  for i,(r,want) in enumerate(zip(receipts,original)):
   assert r==dict(step=i+1,tensor_sha256=want['phase_sha256'],receipt=want['original_receipt'])
  a=np.load(c['arrays_path'],allow_pickle=False);assert set(a.files)=={'logits','probabilities','labels','weights'}
  assert all(a[k].shape==(100,128) and np.isfinite(a[k]).all() for k in a.files)
  z=a['logits'].astype(np.float64).ravel();p=a['probabilities'].astype(np.float64).ravel();y=a['labels'].ravel();w=a['weights'].ravel()
  assert np.all(np.sum(a['labels']==1,axis=1)==32) and np.all(np.sum(a['labels']==0,axis=1)==96)
  assert np.array_equal(w,np.where(y==1,1,20))
  predicted=np.exp(-np.logaddexp(0,-z));gap=float(np.max(abs(predicted-p)));assert gap<=1e-7
  identity=(a['labels'].tobytes(),a['weights'].tobytes(),c['receipts_sha256'])
  if c['target'] in matched:assert identity==matched[c['target']]
  else:matched[c['target']]=identity
  positive=z[y==1];negative=np.sort(z[y==0]);auc=float(np.mean((np.searchsorted(negative,positive,side='left')+np.searchsorted(negative,positive,side='right'))/2)/len(negative))
  bce=np.logaddexp(0,z)-y*z
  groups={}
  for label in [0,1]:
   mask=y==label
   groups[str(label)]=dict(rows=int(mask.sum()),logit_quantiles=np.quantile(z[mask],[0,.05,.5,.95,1]).tolist(),probability_quantiles=np.quantile(p[mask],[0,.05,.5,.95,1]).tolist(),probability_ge_half=int(np.count_nonzero(p[mask]>=.5)),mean_unweighted_bce=float(np.mean(bce[mask])))
  p0=32/(32+96*20);constant=-(32*np.log(p0)+96*20*np.log1p(-p0))/128
  summaries.append(dict(target=c['target'],condition=c['condition'],arrays_sha256=c['arrays_sha256'],receipts_sha256=c['receipts_sha256'],classes=groups,empirical_auc=auc,weighted_mean_bce=float(np.mean(w*bce)),constant_optimal_probability=p0,constant_optimal_weighted_bce=float(constant),saved_probability_max_error=gap))
 out=dict(source_result_sha256=sha(RESULT),execution_sha256=sha(lock),auditor_sha256=sha(__file__),all_passed=True,summaries=summaries,quantile_order=[0,.05,.5,.95,1],limits='First100 batches only: seen correlated training draws, not full training fit, waveform events, source-independent evaluation or generalization. Receipt hashes authenticated; tensor/model/repeat execution is original runtime evidence, not a second inference. No threshold selection or optimizer update; unchanged promotion gates/freshdata/v30.')
 with OUT.open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
