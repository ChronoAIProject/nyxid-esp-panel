"""Check beam decoder blank remapping against exhaustive CTC path sums."""
import itertools
import unittest

import numpy as np

from audit_phone_ctc_readout import beam_sequences
from phone_sequences import collapse


class BeamReadoutTest(unittest.TestCase):
    def test_blank_first_remapping_repeats_and_batched_sequence_order(self):
        logits = np.array([[[.4, .1, -.2], [-.3, .2, .7], [.1, .9, .2]],
                           [[0, 8, -8], [8, 0, -8], [0, 8, -8]]], np.float32)
        expected = []
        for row in logits:
            p = np.exp(row-row.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            strings = {}
            for path in itertools.product(range(3), repeat=3):
                target = tuple(collapse(path))
                strings[target] = strings.get(target, 0)+float(np.prod([p[i, token] for i, token in enumerate(path)]))
            expected.append(list(max(strings, key=strings.get)))
        self.assertEqual(expected[1], [1, 1])
        self.assertEqual(beam_sequences(logits, width=32), expected)


if __name__ == '__main__':
    unittest.main()
