"""Independent NumPy derivative check for the proposed rendering consistency.

Implements the exact max(norm,1e-8) convention and both-view gradients for
two IDs, four phrases and quiet-to-four-renderings. No model/dataset is opened,
and no coefficient, optimizer update or effectiveness claim is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FLOOR = 1e-8


def objective_and_gradient(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 4 and x.shape[:3] == (2, 4, 5)
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    divisor = np.maximum(norm, FLOOR)
    unit = x / divisor
    loss = np.mean(1 - np.sum(unit[:, :, :1] * unit[:, :, 1:], axis=-1))
    unit_gradient = np.empty_like(unit)
    unit_gradient[:, :, 0] = -np.sum(unit[:, :, 1:], axis=2) / 32
    unit_gradient[:, :, 1:] = -unit[:, :, :1] / 32
    gradient = unit_gradient / divisor
    active = (norm > FLOOR)[..., 0]
    # The derivative of the divisor vanishes below the floor. The boundary is
    # intentionally excluded from this differentiability check.
    gradient[active] -= (unit * np.sum(unit * unit_gradient, axis=-1, keepdims=True)
                         / divisor)[active]
    return float(loss), gradient


def finite_difference(x, step):
    gradient = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        left, right = x.copy(), x.copy()
        left[index] -= step
        right[index] += step
        gradient[index] = (objective_and_gradient(right)[0] -
                           objective_and_gradient(left)[0]) / (2 * step)
    return gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh result required")
    x = np.random.default_rng(726).normal(size=(2, 4, 5, 3))
    loss, gradient = objective_and_gradient(x)
    numerical = finite_difference(x, 1e-6)
    np.testing.assert_allclose(gradient, numerical, atol=2e-9, rtol=2e-6)
    # IDs and phrases can be permuted, but no different phrase/ID is ever
    # introduced as a positive. Per-view positive rescaling is invariant only
    # when every norm stays above the documented floor.
    permutation = x[::-1, ::-1]
    np.testing.assert_allclose(objective_and_gradient(permutation)[0], loss, atol=1e-15)
    rescaled = x * np.arange(1, 41).reshape(2, 4, 5, 1)
    np.testing.assert_allclose(objective_and_gradient(rescaled)[0], loss, atol=1e-15)
    # Changing one phrase leaves every other phrase's gradient unchanged.
    changed = x.copy()
    changed[0, 1, 2] *= -1
    altered_gradient = objective_and_gradient(changed)[1]
    keep = np.ones((2, 4), dtype=bool)
    keep[0, 1] = False
    np.testing.assert_array_equal(gradient[keep], altered_gradient[keep])
    zero = np.zeros_like(x)
    zero_loss, zero_gradient = objective_and_gradient(zero)
    assert zero_loss == 1 and np.count_nonzero(zero_gradient) == 0
    collapsed = np.zeros_like(x)
    collapsed[..., 0] = 1
    collapsed_loss, collapsed_gradient = objective_and_gradient(collapsed)
    assert collapsed_loss == 0 and np.count_nonzero(collapsed_gradient) == 0
    tiny = np.random.default_rng(727).normal(size=x.shape) * 1e-10
    tiny_loss, tiny_gradient = objective_and_gradient(tiny)
    tiny_numerical = finite_difference(tiny, 1e-13)
    np.testing.assert_allclose(tiny_gradient, tiny_numerical, atol=.002, rtol=2e-6)
    # One quiet view is zero while its rendered views are not. The exact floor
    # convention produces large finite gradients; it does not erase that pair.
    partial_zero = x.copy()
    partial_zero[0, 0, 0] = 0
    partial_loss, partial_gradient = objective_and_gradient(partial_zero)
    assert np.isfinite(partial_gradient).all()
    assert np.linalg.norm(partial_gradient[0, 0, 0]) > 1e5
    # Complete collapse can be stationary for BCE plus D as well: zero head,
    # identical e, and the weighted class-prior bias. Do not claim a guarantee.
    labels = np.zeros((2, 4, 5))
    labels[:, 0] = 1
    weights = np.where(labels == 1, 1., 20.)
    prior = np.sum(labels * weights) / weights.sum()
    bias = np.log(prior / (1 - prior))
    p = 1 / (1 + np.exp(-bias))
    logit_gradient = weights * (p - labels) / labels.size
    bias_gradient = float(logit_gradient.sum())
    weight_gradient = np.sum(logit_gradient[..., None] * collapsed, axis=(0, 1, 2))
    np.testing.assert_allclose(bias_gradient, 0., atol=1e-15)
    np.testing.assert_allclose(weight_gradient, 0., atol=1e-15)
    # With a nonzero head, opposite binary labels provide different embedding
    # gradients even for identical embeddings. This finite example is not a
    # general noncollapse or optimization-convergence proof.
    head = np.array([.3, -.2, .5])
    p = 1 / (1 + np.exp(-head[0]))
    positive_gradient = (p - 1) * head
    negative_gradient = 20 * p * head
    assert positive_gradient @ negative_gradient < 0
    result = dict(scope=__doc__, recipe_commit="3c075f5", floor=FLOOR,
                  pairs=32, shape=list(x.shape), example_inputs=x.tolist(),
                  example_loss=loss, analytic_gradient=gradient.tolist(),
                  gradient_norm=float(np.linalg.norm(gradient)),
                  maximum_finite_difference_error=float(np.max(np.abs(gradient - numerical))),
                  phrase_id_permutation_invariant=True, above_floor_rescaling_invariant=True,
                  no_cross_phrase_gradient_coupling=True,
                  zero_vectors=dict(loss=zero_loss, gradient_norm=float(np.linalg.norm(zero_gradient))),
                  constant_unit_vectors=dict(loss=collapsed_loss, gradient_norm=float(np.linalg.norm(collapsed_gradient))),
                  tiny_vectors=dict(loss=tiny_loss, gradient_norm=float(np.linalg.norm(tiny_gradient)),
                                    maximum_finite_difference_error=float(np.max(np.abs(tiny_gradient - tiny_numerical)))),
                  one_zero_quiet_view=dict(loss=partial_loss,
                                           quiet_gradient_norm=float(np.linalg.norm(partial_gradient[0, 0, 0]))),
                  stationary_bce_collapse=dict(weighted_prior=float(prior), bias=float(bias),
                                              bias_gradient=bias_gradient, weight_gradient=weight_gradient.tolist(),
                                              embedding_gradient_zero_because_head_zero=True),
                  nonzero_head_opposite_label_gradients=dict(positive=positive_gradient.tolist(),
                                                             negative=negative_gradient.tolist()),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limits="Pure NumPy formula checks and counterexamples. No original-network connectivity, "
                         "BN invariance, real initialization gradient scale, matched sampler, chosen lambda, "
                         "training improvement or acceptance evidence.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("example_loss", "gradient_norm", "maximum_finite_difference_error")}, indent=2))


if __name__ == "__main__":
    main()
