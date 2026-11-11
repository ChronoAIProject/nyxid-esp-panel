"""Training-only diagnostic: single-window fit versus five-score support."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from evaluate import Detector, smooth

root=Path('.local/wakeword/model-gpu-v37-ami-extra')
pack=Path('.local/wakeword/british-training-features-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(pack/'training.npz')=='c0c389fc4a22e1fb9807d99e8a7a98e0e028696d2047dd27dce81f1de8fef096'
assert sha(root/'hey_nyx.tflite')=='09c5191e9b8a1ebe5ebfdfd62e22ea49f520d0eae4fe826910ec614187705d06'
training=json.loads((root/'training.json').read_text())
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
p=argparse.ArgumentParser();mixednet.model_parameters(p)
model=mixednet.model(p.parse_args(training['architecture_args']),(150,40),None)
model.load_weights(root/'best.weights.h5')
z=np.load(pack/'training.npz',allow_pickle=False);x=z['x'];y=z['y']
scores=np.concatenate([model(np.clip(x[i:i+64].astype(np.float32)/25.6,0,26),training=False).numpy().reshape(-1) for i in range(0,len(x),64)])
records=[r for r in json.loads((pack/'report.json').read_text())['records'] if r['pack_index'] is not None]
assert len(records)==len(x)==len(scores)
detector=Detector(root/'hey_nyx.tflite');details=[];arrays={}
for i,row in enumerate(records):
    detector.reset();q=detector.predict(x[i]);arrays[f'stream_{i}']=q
    details.append({'index':i,'source_group':row['source_group'],'voice':row['voice'],'source':row['source'],
        'target':row['target'],'label':int(y[i]),'variant':row['variant'],'window_float':float(scores[i]),
        'stream_last':float(q[-1]),'stream_max':float(max(q)),
        'stream_max_five_mean':float(max(smooth(q))),
        'score_sha256':hashlib.sha256(q.tobytes()).hexdigest()})
    if i%100==0:print(json.dumps({'completed':i+1,'total':len(x)}),flush=True)
out=Path('.local/wakeword/v37-training-support-probe-v1');out.mkdir(exist_ok=False)
np.savez_compressed(out/'scores.npz',window=scores,**arrays)
summary={}
for label in (0,1):
    group=[d for d in details if d['label']==label]
    summary[str(label)]={'examples':len(group)}
    for key in ('window_float','stream_last','stream_max','stream_max_five_mean'):
        summary[str(label)][key]={'above_half':sum(d[key]>=.5 for d in group),
            'above_historical_threshold':sum(d[key]>=.99609375 for d in group)}
result={'scope':'Training-only isolated150-feature crop probe, not PCM streaming accuracy or new acceptance.',
    'model_sha256':sha(root/'hey_nyx.tflite'),'pack_sha256':sha(pack/'training.npz'),
    'weights_sha256':sha(root/'best.weights.h5'),'script_sha256':sha(Path(__file__)),
    'scores_sha256':sha(out/'scores.npz'),'tensorflow':tf.__version__,
    'summary':summary,'details':details,
    'limits':'Training crops reused descriptively. Stream begins from zero model state at crop start, without original frontend history; no padding invented. Comparing offline float window and INT8 streaming also includes quantization/state/phase differences. Not an isolated causal test; no new labels/thresholds/training or final data.'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
