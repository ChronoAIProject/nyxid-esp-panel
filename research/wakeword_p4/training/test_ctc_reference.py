"""Exhaustive path and finite-difference checks of the diagnostic CTC reference."""
import itertools
import unittest

import numpy as np

from ctc_reference import loss_and_gradient


class ReferenceTest(unittest.TestCase):
    def test_exhaustive_path_probability_and_posterior_gradient(self):
        x = np.random.default_rng(51).normal(size=(5, 3))
        p = np.exp(x-x.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        for target in ([1, 2, 1], [1, 1], [2]):
            mass = 0.
            occupancy = np.zeros_like(p)
            for path in itertools.product(range(3), repeat=len(x)):
                collapsed = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]
                if collapsed == target:
                    weight = np.prod([p[t, v] for t, v in enumerate(path)])
                    mass += weight
                    for t, v in enumerate(path):
                        occupancy[t, v] += weight
            loss, gradient = loss_and_gradient(x, target)
            self.assertAlmostEqual(loss, -np.log(mass), places=13)
            np.testing.assert_allclose(gradient, p-occupancy/mass, atol=1e-14, rtol=0)

    def test_large_logits_finite_differences_and_probability_invariance(self):
        x = np.random.default_rng(52).normal(size=(8, 4))*20
        loss, gradient = loss_and_gradient(x, [1, 2, 3, 2])
        for time, token in ((0, 1), (2, 2), (4, 3), (7, 0)):
            hi, lo = x.copy(), x.copy()
            hi[time, token] += 1e-4; lo[time, token] -= 1e-4
            numerical = (loss_and_gradient(hi, [1, 2, 3, 2])[0]
                         -loss_and_gradient(lo, [1, 2, 3, 2])[0])/2e-4
            self.assertAlmostEqual(gradient[time, token], numerical, places=8)
        shifted_loss, shifted_gradient = loss_and_gradient(x+np.arange(8)[:, None]*100, [1, 2, 3, 2])
        self.assertAlmostEqual(loss, shifted_loss, places=11)
        np.testing.assert_allclose(gradient, shifted_gradient, atol=1e-12, rtol=0)

    def test_rejects_impossible_or_invalid_targets(self):
        for x, target in ((np.zeros((2, 3)), [1, 1]), (np.zeros((4, 3)), []),
                          (np.zeros((4, 3)), [0]), (np.zeros((4, 3)), [3]),
                          (np.full((4, 3), np.nan), [1])):
            with self.assertRaises(ValueError):
                loss_and_gradient(x, target)


if __name__ == '__main__':
    unittest.main()
