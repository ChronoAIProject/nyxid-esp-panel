"""Keep mining ablations comparable without touching real held-out audio."""
import unittest

import numpy as np

from train_head import Sampler


class MiningComparabilityTest(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for label, kind, engine, base in ((1, 'tts', 'a', 0), (1, 'tts', 'b', 10),
                                          (0, 'tts', 'a', 20), (0, 'tts', 'b', 30),
                                          (0, 'speech', 'human', 40)):
            self.rows.append({'id': str(base), 'label': label, 'kind': kind, 'engine': engine,
                              'split': 'training', 'features_sha256': str(base), 'frames': 60,
                              'keyword_end': 2.4,
                              'array': np.tile(np.arange(60)[:, None] + base, (1, 96))})
        self.mined = [{'id': row['id'], 'end': 59, 'feature_sha256': row['features_sha256']}
                      for row in self.rows if not row['label']]

    def test_quarter_mining_keeps_labels_positives_and_remaining_draws(self):
        control = Sampler(self.rows, 727)
        treatment = Sampler(self.rows, 727, self.mined, hard_negative_stride=4)
        for _ in range(10):
            a, labels = control.batch()
            b, other_labels = treatment.batch()
            np.testing.assert_array_equal(labels, other_labels)
            positive = labels[:, 0].astype(bool)
            np.testing.assert_array_equal(a[positive], b[positive])
            changed = np.any(a != b, axis=(1, 2, 3))
            self.assertGreater(changed.sum(), 10)
            self.assertLessEqual(changed.sum(), 24)

    def test_default_retains_half_mining(self):
        default = Sampler(self.rows, 727, self.mined)
        explicit = Sampler(self.rows, 727, self.mined, hard_negative_stride=2)
        for _ in range(3):
            a, y = default.batch()
            b, z = explicit.batch()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(y, z)

    def test_invalid_stride_and_heldout_mining_rejected(self):
        for value in (True, 0, 1, 2.5):
            with self.assertRaises(ValueError):
                Sampler(self.rows, 727, self.mined, hard_negative_stride=value)
        self.rows[-1]['split'] = 'validation'
        with self.assertRaises(ValueError):
            Sampler(self.rows, 727, self.mined)


if __name__ == '__main__':
    unittest.main()
