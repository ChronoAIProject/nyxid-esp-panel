"""Verify stable five-probability-mean BCE and actual checkpoint gradients."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from sequence_loss import build_sequence_logits, mean_probability_bce
from sequence_scorer import build_sequence_scorer
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
extremes=[]
for value,label in [(20.,0.),(-20.,1.)]:
    z=tf.Variable([[value]*5],dtype=tf.float32)
    with tf.GradientTape() as tape:l=mean_probability_bce(tf.constant([label]),z)
    g=tape.gradient(l,z).numpy();assert np.isfinite(g).all()
    np.testing.assert_allclose(g,np.full((1,5),.2 if label==0 else -.2),atol=1e-6)
    extremes.append({'logit':value,'label':label,'loss':float(l[0]),'gradient':g.tolist()})
root=Path('.local/wakeword/model-gpu-v37-ami-extra')
pack=Path('.local/wakeword/british-sequence-training-features-v1/training.npz')
assert sha(pack)=='e5e731ff9fe390100b58d153002648cc238edf6e7daefcc58ea39929d256efc7'
r=json.loads((root/'training.json').read_text())
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
p=argparse.ArgumentParser();mixednet.model_parameters(p)
window=mixednet.model(p.parse_args(r['architecture_args']),(150,40),None)
window.load_weights(root/'best.weights.h5')
for layer in window.layers:
    if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
logits=build_sequence_logits(window);probabilities=build_sequence_scorer(window)
z=np.load(pack,allow_pickle=False);x=np.clip(z['x'][:16].astype(np.float32)/25.6,0,26);y=z['y'][:16]
before=[v.numpy().copy() for v in window.variables]
with tf.GradientTape() as tape:
    l=logits(x,training=True);assert l.shape==(16,5)
    loss=tf.reduce_mean(mean_probability_bce(y,l))
g=tape.gradient(loss,window.trainable_variables)
np.testing.assert_allclose(tf.sigmoid(l).numpy(),probabilities(x,training=False).numpy(),atol=1e-6)
# Independent direct-window logits via original shared Flatten input and Dense.
trunk=tf.keras.Model(window.input,window.layers[-2].output);dense=window.layers[-1]
with tf.GradientTape() as tape:
    values=[]
    for at in range(0,13,3):
        h=trunk(x[:,at:at+150],training=False)
        values.append(tf.reshape(tf.matmul(h,dense.kernel)+dense.bias,[-1]))
    direct=tf.stack(values,axis=1)
    direct_loss=tf.reduce_mean(mean_probability_bce(y,direct))
dg=tape.gradient(direct_loss,window.trainable_variables)
errors=[]
for a,b in zip(g,dg,strict=True):
    assert a is not None and b is not None
    assert np.isfinite(a.numpy()).all() and np.isfinite(b.numpy()).all()
    np.testing.assert_allclose(a.numpy(),b.numpy(),rtol=2e-3,atol=2e-5)
    errors.append(float(np.max(np.abs(a.numpy()-b.numpy()))))
for v,b in zip(window.variables,before,strict=True):np.testing.assert_array_equal(v.numpy(),b)
out={'scope':'Training-only real-v37 gradient equivalence; optimizer0, no acceptance.',
    'pack_sha256':sha(pack),'weights_sha256':sha(root/'best.weights.h5'),
    'sequence_loss_origin_commit':'e32f0d6','sequence_loss_sha256':sha(Path(__file__).resolve().parents[1]/'training/sequence_loss.py'),
    'script_sha256':sha(Path(__file__)),'extremes':extremes,'examples':16,
    'loss':float(loss),'direct_loss':float(direct_loss),'gradient_variables':len(g),
    'max_gradient_error':max(errors),'gradient_l1':sum(float(tf.reduce_sum(tf.abs(a))) for a in g),
    'all_variables_unchanged':True,'sigmoid_logits_matches_probability_scorer':True}
Path('research/wakeword_p4/results/4090-v39-gradient-probe-v1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
