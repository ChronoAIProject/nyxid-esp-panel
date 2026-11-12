"""Fixed first saved batch, certified initialization, deterministic full FP32."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
p = argparse.ArgumentParser(); p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
p.add_argument('--output', type=Path, required=True); a = p.parse_args(); assert not a.output.exists()
if a.device == 'cpu': tf.config.set_visible_devices([], 'GPU')
else: assert tf.config.get_visible_devices('GPU')
tf.config.experimental.enable_op_determinism()
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(726)
base = Path('.local/wakeword'); sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sys.path.insert(0, str((base/'microWakeWord').resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from microwakeword import mixednet
from batchnorm_policy import apply_policy
from rendering_consistency import rendering_consistency
training = json.loads((base/'model-paired-control-smoke-v1/training.json').read_text())
arch = argparse.ArgumentParser(); mixednet.model_parameters(arch)
model = mixednet.model(arch.parse_args(training['architecture_args']), (150, 40), None)
snapshot = base/'derived-probe-initialization-v1/derived-probe-initialization.weights.h5'
model.load_weights(snapshot); apply_policy(model, 'frozen')
variables = model.trainable_variables; head = model.layers[-1]
extractor = tf.keras.Model(model.inputs, head.input)
batches = base/'paired-consistency-init-gradient-probe-v1/probe-batches.npz'
assert sha(batches) == '7d781dfc435417632aaa2c8995d6a955772b29b3ca3f88fbfd87d57108e994cb'
with np.load(batches) as z: values = [z[k][0] for k in ['main_x', 'main_y', 'main_weight', 'pair_x', 'pair_y', 'pair_weight']]
before = [v.numpy().copy() for v in model.variables]
@tf.function
def infer(x, y, w, sx, sy, sw):
    with tf.GradientTape(persistent=True) as tape:
        e = extractor(x, training=True); se = extractor(sx, training=True)
        logits = tf.matmul(e, head.kernel)+head.bias
        sl = tf.matmul(se, head.kernel)+head.bias
        main = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)[:, 0]*w)
        pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sy, logits=sl)[:, 0]*sw)
        supervised = main+pair
        d = rendering_consistency(tf.reshape(se, [2, 4, 5, 300]))
    gs = tape.gradient(supervised, variables); gd = tape.gradient(d, variables)
    assert all(g is not None for g in gs)
    gd = [tf.zeros_like(v) if g is None else g for v, g in zip(variables, gd)]
    return tf.stack([main, pair, d]), gs, gd
rows = []
for repeat in range(2):
    metrics, gs, gd = infer(*values)
    packed = dict(metrics=metrics.numpy(), supervised=np.concatenate([g.numpy().reshape(-1) for g in gs]),
                  consistency=np.concatenate([g.numpy().reshape(-1) for g in gd]))
    assert all(np.isfinite(v).all() for v in packed.values())
    rows.append(packed)
for k in rows[0]: np.testing.assert_array_equal(rows[0][k], rows[1][k])
for x, v in zip(before, model.variables): np.testing.assert_array_equal(x, v.numpy())
a.output.mkdir(); np.savez_compressed(a.output/'arrays.npz', **rows[0])
r = dict(device=a.device, determinism_enabled=True, tf32_enabled=False, exact_repeat=True,
    no_weight_updates=True, snapshot_sha256=sha(snapshot), batches_sha256=sha(batches),
    metrics=rows[0]['metrics'].tolist(), arrays_sha256=sha(a.output/'arrays.npz'),
    script_sha256=sha(Path(__file__)),
    limits='First saved batch only. New explicit numerical execution policy, not retroactive passage of original nondeterministic GPU check or new coefficient estimate.')
(a.output/'report.json').write_text(json.dumps(r, indent=2)+'\n'); print(json.dumps(r))
