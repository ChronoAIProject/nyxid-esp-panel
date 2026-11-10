"""Inspect raw positive scores on deterministic development subsets; not event accuracy."""
import sys,json,hashlib
from pathlib import Path
import numpy as np, torch
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from train_embedding_head import head
p=Path('.local/wakeword/embedding-development-v1'); rows=json.loads((p/'index.json').read_text()); raw=np.memmap(p/'embeddings.bin',dtype='<f4',mode='r').reshape(-1,96)
m=head().cuda(); m.load_state_dict(torch.load('.local/wakeword/embedding-head-v1b/best.pt',weights_only=True));m.eval()
out=[]
for split in ['training','validation']:
 for engine in sorted({x.get('engine') for x in rows if x['kind']=='generated'}):
  r=[x for x in rows if x['kind']=='generated' and x['label']==1 and x['split']==split and x['engine']==engine]
  if not r:continue
  r=sorted(r,key=lambda x:hashlib.sha256(str(x).encode()).hexdigest())[:300]
  sampled=[];peak=[];ends=[]
  for x in r:
   z=raw[x['offset']:x['offset']+x['frames']]; windows=np.lib.stride_tricks.sliding_window_view(z,16,axis=0).transpose(0,2,1)
   with torch.inference_mode():scores=m(torch.tensor(np.array(windows),device='cuda')).sigmoid().flatten().cpu().numpy()
   e=x['keyword_end_seconds']; i=int(np.ceil(e/.08)); sampled.extend(scores[i:i+3]);t=(np.arange(len(scores))+1)*.08
   good=(t>=e-.1)&(t<=e+.8);peak.append(float(max(scores[good])));ends.append(e)
  out.append(dict(split=split,engine=engine,n=len(r),sampled_score_quantiles=np.quantile(sampled,[0,.1,.5,.9,1]).tolist(),hit_window_peak_quantiles=np.quantile(peak,[0,.1,.5,.9,1]).tolist(),end_seconds_quantiles=np.quantile(ends,[0,.5,1]).tolist()))
print(json.dumps({'role':'Development-only raw positive window diagnostic; not event recall', 'subset':'Up to 300 positive rendering rows per split/engine, sorted by SHA256(str(row)); correlated variants', 'model_sha256':hashlib.sha256(Path('.local/wakeword/embedding-head-v1b/best.pt').read_bytes()).hexdigest(), 'cache_index_sha256':hashlib.sha256((p/'index.json').read_bytes()).hexdigest(), 'quantile_levels':[0,.1,.5,.9,1], 'statistics':out},indent=2))
