"""Synthetic finite-difference compatibility only, no model data or updates.

Target anchors only; positives = other target examples; denominator = every
nonself example including Unknown. Unknown examples are never mutual positives.
Mean over anchors and over each anchor's positives, temperature fixed at 1 for
this mathematical example, not a proposed training coefficient/temperature.
Compare to target-only denominator, which degenerates with two target examples.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

def loss(x, labels, include_unknown):
    norms = np.linalg.norm(x, axis=1)
    if np.any(norms == 0) or np.count_nonzero(labels) < 2:
        raise ValueError('Requires nonzero embeddings and at least two targets')
    u = x/norms[:, None]
    values = []
    for anchor in np.flatnonzero(labels):
        positive = np.flatnonzero(labels & (np.arange(len(labels)) != anchor))
        denominator = np.flatnonzero((np.ones(len(labels), dtype=bool) if include_unknown else labels)
                                    & (np.arange(len(labels)) != anchor))
        scores = u[denominator]@u[anchor]
        top = scores.max()
        logsumexp = top+np.log(np.exp(scores-top).sum())
        values.append(logsumexp-np.mean(u[positive]@u[anchor]))
    return float(np.mean(values))

def gradient(x, labels, include_unknown):
    h = 1e-5
    result = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += h; minus[index] -= h
        result[index] = (loss(plus, labels, include_unknown)-loss(minus, labels, include_unknown))/(2*h)
    return result

def analytic_gradient(x, labels):
    norms = np.linalg.norm(x, axis=1)
    u = x/norms[:, None]; g = np.zeros_like(x)
    anchors = np.flatnonzero(labels)
    for anchor in anchors:
        indices = np.flatnonzero(np.arange(len(labels)) != anchor)
        scores = u[indices]@u[anchor]
        coefficients = np.exp(scores-scores.max())
        coefficients /= coefficients.sum()
        coefficients -= labels[indices]/np.sum(labels[indices])
        coefficients /= len(anchors)
        g[anchor] += coefficients@u[indices]
        g[indices] += coefficients[:, None]*u[anchor]
    return (g-u*np.sum(g*u, axis=1, keepdims=True))/norms[:, None]

labels = np.array([True, True, False, False])
x = np.array([[1., 0.], [.5, np.sqrt(.75)], [-1., .2], [.2, -1.]])
rows = {}
for include in [False, True]:
    g = gradient(x, labels, include)
    assert np.isfinite(g).all()
    rows['all_nonself' if include else 'target_only'] = dict(
        loss=loss(x, labels, include), finite_difference_gradient=g.tolist(), gradient_norm=float(np.linalg.norm(g)))
assert rows['target_only']['loss'] == 0 and rows['target_only']['gradient_norm'] == 0
assert rows['all_nonself']['gradient_norm'] > .1
analytic = analytic_gradient(x, labels)
np.testing.assert_allclose(analytic, gradient(x, labels, True), atol=1e-9, rtol=1e-7)
permutation = [1, 0, 3, 2]
np.testing.assert_allclose(loss(x, labels, True), loss(x[permutation], labels[permutation], True), atol=1e-14)
np.testing.assert_allclose(loss(x, labels, True), loss(x*np.array([2, 3, 4, 5])[:, None], labels, True), atol=1e-14)
collapsed = np.tile([1., 0.], (4, 1))
collapse_gradient = gradient(collapsed, labels, True)
assert np.linalg.norm(collapse_gradient) == 0
guards = {}
for name, a, y in [('zero_norm', np.zeros((4, 2)), labels),
                    ('one_target', x, np.array([True, False, False, False]))]:
    try:
        loss(a, y, True)
    except ValueError:
        guards[name] = True
    else:
        raise AssertionError(name)
result = dict(scope='Pure synthetic single-keyword loss compatibility, not selection or model training',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs=x.tolist(), labels=labels.tolist(),
    temperature_example_only=1., examples=rows, permutation_invariant=True, positive_rescaling_invariant=True,
    analytic_gradient=analytic.tolist(), analytic_finite_difference_match=True,
    guards=guards, collapsed_loss=loss(collapsed, labels, True), collapsed_gradient_norm=float(np.linalg.norm(collapse_gradient)),
    limits='All-nonself denominator avoids the two-target identically-zero defect, but complete normalized collapse remains stationary. Unknowns are not positive pairs. No convergence, anti-collapse, margin, FAR, augmentation or training-effectiveness claim; no loss/temperature/weight selected.')
Path('research/wakeword_p4/results/4090-single-keyword-contrastive-compatibility-v1.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
