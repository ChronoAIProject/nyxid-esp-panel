"""Audit Mac sequence helper against actual v37 saved training streams."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from sequence_scorer import build_sequence_scorer
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
root=Path('.local/wakeword/model-gpu-v37-ami-extra')
history=Path('.local/wakeword/v37-full-history-probe-v1')
assert sha(history/'arrays.npz')=='bfc595c6b2fab777727017bcfc495ffc84cf13907f0fcb2ccb6dfdf008b15ec6'
report=json.loads((history/'report.json').read_text());training=json.loads((root/'training.json').read_text())
assert sha(root/'hey_nyx.tflite')==report['model_sha256']
p=argparse.ArgumentParser();mixednet.model_parameters(p)
window=mixednet.model(p.parse_args(training['architecture_args']),(150,40),None)
window.load_weights(root/'best.weights.h5')
for layer in window.layers:
    if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
sequence=build_sequence_scorer(window)
z=np.load(history/'arrays.npz',allow_pickle=False);details=[]
before=[v.numpy().copy() for v in window.variables]
for row in report['details']:
    i=row['index'];raw=z[f'raw_{i}'];n=len(raw)//3*3
    x=np.clip(raw[:n].astype(np.float32)/25.6,0,26)[None]
    result=sequence(x[:,1:],training=False).numpy()[0]
    reference=z[f'float_{i}'][49:]
    assert result.shape==reference.shape
    aligned=z[f'aligned_{i}'];compared=min(len(aligned),len(result))
    details.append({'index':i,'target':row['target'],
        'sequence_stream_max_error':float(np.max(np.abs(result-reference))),
        'sequence_window_max_error':float(np.max(np.abs(result[:compared]-aligned[:compared]))),
        'outputs':len(result)})
    if i%200==0:print(json.dumps({'completed':i+1}),flush=True)
for v,b in zip(window.variables,before,strict=True):np.testing.assert_array_equal(v.numpy(),b)
r={'scope':'Actual v37 weights/1002 saved training streams only; no optimizer/new data/acceptance.',
    'helper_origin_commit':'3baa61e','helper_sha256':sha(Path(__file__).resolve().parents[1]/'training/sequence_scorer.py'),
    'script_sha256':sha(Path(__file__)),'model_sha256':report['model_sha256'],
    'history_arrays_sha256':sha(history/'arrays.npz'),'weights_sha256':sha(root/'best.weights.h5'),
    'sources':len(details),'compared_outputs':sum(d['outputs'] for d in details),
    'sequence_stream_max_error':max(d['sequence_stream_max_error'] for d in details),
    'sequence_window_max_error':max(d['sequence_window_max_error'] for d in details),
    'all_variables_unchanged':True,'parameters':sequence.count_params(),'tensorflow':tf.__version__,
    'details':details,'limits':'Normalization clip(raw/25.6,0,26), phase1 and mature output49 onward. No tolerance hidden, no deployment parity/gradient/training improvement claim. Saved reference stream was computed on GPU; helper here CPU, so backend numeric differences retained.'}
out=Path('research/wakeword_p4/results/4090-v37-sequence-mapping-v1.json')
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='details'}),flush=True)
