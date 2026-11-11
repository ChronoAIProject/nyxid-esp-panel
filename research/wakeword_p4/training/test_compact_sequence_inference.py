"""Independent exhaustive paths for the inference CTC probability."""
import itertools
import unittest

import numpy as np

from compact_sequence_inference import target_log_probability
from phone_sequences import collapse


class SequenceProbabilityTest(unittest.TestCase):
    def test_all_paths_match_absolute_probability_without_forced_choice(self):
        rng = np.random.default_rng(1)
        logits = rng.normal(size=(2, 4, 6))
        wanted = []
        for item in logits:
            p = np.exp(item-item.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            total = sum(np.prod([p[i, token] for i, token in enumerate(path)])
                        for path in itertools.product(range(6), repeat=4) if collapse(path) == [1, 2])
            wanted.append(total)
        actual = np.exp(target_log_probability(logits))
        np.testing.assert_allclose(actual, wanted, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(target_log_probability(logits+123), np.log(wanted), atol=1e-12)
        self.assertTrue(np.all((actual >= 0) & (actual <= 1)))

    def test_order_blank_and_out_of_grammar_mass_remain_distinct(self):
        logits = np.full((1, 4, 6), -20.)
        logits[0, np.arange(4), [1, 0, 2, 0]] = 20
        self.assertGreater(np.exp(target_log_probability(logits))[0], .999)
        self.assertLess(np.exp(target_log_probability(logits[:, ::-1]))[0], 1e-10)
        for path in ([0, 0, 0, 0], [1, 0, 3, 0], [1, 2, 1, 2]):
            logits[:] = -20
            logits[0, np.arange(4), path] = 20
            self.assertLess(np.exp(target_log_probability(logits))[0], 1e-10)
        self.assertEqual(target_log_probability(np.zeros((1, 0, 6)))[0], -np.inf)
        for bad in (np.zeros((2, 6)), np.zeros((1, 4, 5)), np.full((1, 4, 6), np.nan)):
            with self.assertRaises(ValueError):
                target_log_probability(bad)


if __name__ == '__main__':
    unittest.main()
