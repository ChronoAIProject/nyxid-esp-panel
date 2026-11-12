"""Exact324 US supervised crops, window float only; no optimization."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/model-gpu-v44-ami-extra');pack=Path('.local/wakeword/us-training-features-v1');out=Path('.local/wakeword/v44-us-center-fit-v1');out.mkdir(exist_ok=False)
assert sha(root/'hey_nyx.tflite')=='88de16e71b5f49b5e85d226b74a07aa036324dcaab48e6f58c9c8a9343608652'
assert sha(pack/'training.npz')=='4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da'
training=json.loads((root/'training.json').read_text());meta=json.loads((pack/'report.json').read_text());p=argparse.ArgumentParser();mixednet.model_parameters(p);model=mixednet.model(p.parse_args(training['architecture_args']),(150,40),None);model.load_weights(root/'best.weights.h5')
with np.load(pack/'training.npz',allow_pickle=False) as z:x=z['x'];y=z['y']
scores=np.concatenate([model(np.clip(x[a:a+64].astype(np.float32)/25.6,0,26),training=False).numpy().reshape(-1) for a in range(0,324,64)]);assert np.isfinite(scores).all();np.save(out/'scores.npy',scores)
groups={};details=[]
for i,(score,row) in enumerate(zip(scores,meta['records'])):
 assert row['pack_index']==i and y[i]==int(row['target']=='hey_nyx')
 details.append(dict(index=i,voice=row['voice'],target=row['target'],source=row['source'],variant=row['variant'],score=float(score)))
 for key in ['all/'+str(int(y[i])),row['voice']+'/'+row['target']]:
  g=groups.setdefault(key,dict(variants=0,high={str(t):0 for t in [.5,.992,.99609375]}));g['variants']+=1
  for t in [.5,.992,.99609375]:g['high'][str(t)]+=int(score>=t)
r=dict(scope='All324 exact supervised US training crops at selected v44 weights; float window outputs',model_sha256=sha(root/'hey_nyx.tflite'),weights_sha256=sha(root/'best.weights.h5'),pack_sha256=sha(pack/'training.npz'),script_sha256=sha(Path(__file__)),scores_sha256=sha(out/'scores.npy'),groups=groups,details=details,limits='Training fit only, not full-stream event or quantization proof. No optimization, new-development/final data, selection or promotion.')
(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:groups[k] for k in ['all/0','all/1']}))
