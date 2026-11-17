"""Retrospective descriptive objective audit; no fitting or coefficient choice."""
import gzip,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
meta=json.loads((R/'4090-frozen-history-teacher-metadata-math-preflight-v1.json').read_text())['inventory']
out={'schema':'frozen-history-teacher-loss-balance-v1','arms':{},'structural_normalization':{'main_rows':128,'paired_rows':8,'main_group_coefficient':1,'paired_group_coefficient':1,'paired_per_row_relative_to_main':16},'limits':'Retrospective scalar loss magnitudes are not gradient magnitudes, causal attribution, generalization or coefficient selection. Pair labels/masks/history distributions differ from main. No model/threshold change.'}
identities=None
for arm in ['control','teacher']:
 p=Path(f'.local/wakeword/frozen-history-teacher-training-v1/{arm}/batches.jsonl.gz')
 with gzip.open(p,'rt') as f:rows=[json.loads(s) for s in f]
 assert len(rows)==20000 and [r['step'] for r in rows]==list(range(1,20001))
 ids=[r['pair_indices'] for r in rows]
 if identities is None:identities=ids
 else:assert identities==ids
 phases={}
 for name,subset in [('all',rows),('first2000',rows[:2000]),('last2000',rows[-2000:])]:
  a=np.array([[r[k] for k in ['main_bce','paired_bce','kl']] for r in subset]);assert np.isfinite(a).all()
  phases[name]={'steps':len(subset),'mean':dict(zip(['main_bce','paired_bce','kl'],a.mean(0).tolist())),'median':dict(zip(['main_bce','paired_bce','kl'],np.median(a,axis=0).tolist())),'paired_larger_than_main_steps':int((a[:,1]>a[:,0]).sum()),'kl_larger_than_main_steps':int((a[:,2]>a[:,0]).sum())}
 counts=np.bincount(np.array(ids).reshape(-1),minlength=4320);assert (counts>0).all()
 positive=sum(int(counts[i]) for i,m in enumerate(meta) if m['source']['label'])
 out['arms'][arm]={'ledger_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'phases':phases,'pair_draws':int(counts.sum()),'positive_pair_draws':positive,'negative_pair_draws':int(counts.sum())-positive,'source_draw_min':int(counts.min()),'source_draw_max':int(counts.max())}
(R/'4090-frozen-history-teacher-loss-balance-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
