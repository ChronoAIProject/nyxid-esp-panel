"""Enumerate every short CTC path independently of the three-state recurrence."""
import itertools
import unittest

import numpy as np

from compact_sequence_inference import target_log_probability
from keyword_occurrence import log_probability
from phone_sequences import collapse


class KeywordOccurrenceTest(unittest.TestCase):
    def test_exhaustive_paths_with_context_and_ctc_repeats(self):
        rng = np.random.default_rng(32)
        for steps in range(1, 6):
            logits = rng.normal(size=(3, steps, 6))
            shifted = logits-logits.max(-1, keepdims=True)
            p = np.exp(shifted)/np.exp(shifted).sum(-1, keepdims=True)
            expected = np.zeros(3)
            for path in itertools.product(range(6), repeat=steps):
                tokens = collapse(path)
                if any(tokens[i:i+2] == [1, 2] for i in range(len(tokens)-1)):
                    expected += np.prod(p[:, np.arange(steps), path], axis=-1)
            actual = np.exp(log_probability(logits))
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(log_probability(logits+123), log_probability(logits), atol=1e-12)
            self.assertTrue(np.all(actual+1e-14 >= np.exp(target_log_probability(logits))))

    def test_context_order_and_intervening_symbols(self):
        for path, match in [([3, 1, 0, 2, 4], True), ([1, 0, 1, 2], True),
                            ([1, 1, 0, 0, 2, 2], True), ([2, 0, 1], False),
                            ([1, 3, 2], False), ([1, 0, 3, 0, 2], False)]:
            x = np.full((1, len(path), 6), -40.)
            x[0, np.arange(len(path)), path] = 40.
            probability = float(np.exp(log_probability(x))[0])
            self.assertGreater(probability, .999999) if match else self.assertLess(probability, 1e-10)
        x = np.full((1, 5, 6), -40.)
        x[0, np.arange(5), [3, 1, 0, 2, 4]] = 40
        self.assertLess(float(np.exp(target_log_probability(x))[0]), 1e-10)
        self.assertGreater(float(np.exp(log_probability(x))[0]), .999999)

    def test_empty_long_blank_and_invalid_geometry(self):
        self.assertEqual(log_probability(np.zeros((1, 0, 6)))[0], -np.inf)
        for token in [0, 1, 2, 3]:
            x = np.full((1, 67, 6), -40.)
            x[..., token] = 40
            self.assertLess(float(np.exp(log_probability(x))[0]), 1e-10)
        for bad in [np.zeros((2, 6)), np.zeros((1, 4, 5)), np.full((1, 4, 6), np.nan)]:
            with self.assertRaises(ValueError):
                log_probability(bad)


if __name__ == '__main__':
    unittest.main()
