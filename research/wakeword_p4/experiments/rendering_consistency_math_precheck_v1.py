"""Analytic/finite difference/TF gradient checks and explicit collapse limits."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from rendering_consistency import EPSILON, rendering_consistency

def numpy_loss_and_gradient(x):
    g = np.zeros_like(x); value = 0.
    for i in range(2):
        for p in range(4):
            a = x[i, p, 0]; na = max(np.linalg.norm(a), EPSILON)
            for arm in range(1, 5):
                b = x[i, p, arm]; nb = max(np.linalg.norm(b), EPSILON); dot = a@b
                value += (1-dot/(na*nb))/32
                ga = -b/(na*nb); gb = -a/(na*nb)
                if np.linalg.norm(a) > EPSILON: ga += dot*a/(na**3*nb)
                if np.linalg.norm(b) > EPSILON: gb += dot*b/(nb**3*na)
                g[i, p, 0] += ga/32; g[i, p, arm] += gb/32
    return value, g

def tf_value_gradient(x):
    variable = tf.Variable(x, dtype=tf.float64)
    with tf.GradientTape() as tape: loss = rendering_consistency(variable)
    g = tape.gradient(loss, variable).numpy()
    assert np.isfinite(g).all() and np.isfinite(loss.numpy())
    return float(loss.numpy()), g

x = np.random.default_rng(227726).normal(size=(2, 4, 5, 3))
value, analytic = numpy_loss_and_gradient(x); observed, automatic = tf_value_gradient(x)
np.testing.assert_allclose(value, observed, atol=1e-14)
np.testing.assert_allclose(analytic, automatic, atol=1e-14, rtol=1e-12)
finite = np.zeros_like(x); h = 1e-5
for at in np.ndindex(x.shape):
    plus, minus = x.copy(), x.copy(); plus[at] += h; minus[at] -= h
    finite[at] = (numpy_loss_and_gradient(plus)[0]-numpy_loss_and_gradient(minus)[0])/(2*h)
np.testing.assert_allclose(analytic, finite, atol=1e-9, rtol=1e-6)
special = {}
for name, example in [('equal_nonzero', np.ones_like(x)), ('zero', np.zeros_like(x)),
                      ('tiny', x*1e-10)]:
    loss, grad = tf_value_gradient(example); nv, ng = numpy_loss_and_gradient(example)
    np.testing.assert_allclose(loss, nv, atol=1e-13)
    np.testing.assert_allclose(grad, ng, atol=1e-6, rtol=1e-12)
    special[name] = dict(loss=loss, gradient_norm=float(np.linalg.norm(grad)), finite=True)
assert special['equal_nonzero']['gradient_norm'] < 1e-14
assert special['zero']['loss'] == 1 and special['zero']['gradient_norm'] == 0

# Retaining BCE does not guarantee escape from a constant representation:
# zero head + weighted-prior bias makes both head and embedding stationary.
e = tf.Variable(np.ones((2, 4, 5, 3)), dtype=tf.float64)
w = tf.Variable(np.zeros(3), dtype=tf.float64)
b = tf.Variable(np.log(1/60), dtype=tf.float64)
y = np.zeros((2, 4, 5)); y[:, 0, :] = 1.
labels = tf.constant(y, dtype=tf.float64); weights = tf.where(labels > 0, 1., 20.)
weights = tf.cast(weights, tf.float64)
with tf.GradientTape() as tape:
    logits = tf.reduce_sum(e*w, axis=-1)+b
    bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)*weights)
    total = bce+rendering_consistency(e)
collapse = tape.gradient(total, [e, w, b])
collapse_norms = [float(tf.linalg.norm(g)) for g in collapse]
assert max(collapse_norms) < 1e-12

# With a nonzero head and different supervised labels, embedding gradients differ.
w.assign([1., -.5, .25])
with tf.GradientTape() as tape:
    logits = tf.reduce_sum(e*w, axis=-1)+b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)*weights)
g = tape.gradient(loss, e).numpy()
assert np.dot(g[0, 0, 0], g[0, 1, 0]) < 0
result = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    loss_source_sha256=hashlib.sha256(Path('research/wakeword_p4/training/rendering_consistency.py').read_bytes()).hexdigest(),
    epsilon=EPSILON, loss=value, analytic_tf_max_error=float(np.abs(analytic-automatic).max()),
    analytic_finite_difference_max_error=float(np.abs(analytic-finite).max()), special_cases=special,
    full_constant_bce_consistency_stationary_gradient_norms=collapse_norms,
    nonzero_head_opposite_label_embedding_gradient_dot=float(np.dot(g[0, 0, 0], g[0, 1, 0])),
    limits='Synthetic formula/gradient checks only; not real-model gradient scaling, anti-collapse guarantee or permission to train.')
Path('research/wakeword_p4/results/4090-rendering-consistency-math-precheck-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
