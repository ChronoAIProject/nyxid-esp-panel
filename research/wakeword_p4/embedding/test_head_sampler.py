"""Keep mining ablations comparable without touching real held-out audio."""
import unittest
import copy

import numpy as np

from train_head import Sampler
from training_endpoints import endpoint_map


class MiningComparabilityTest(unittest.TestCase):
    def setUp(self):
        self.rows = []
        for label, kind, engine, base in ((1, 'tts', 'a', 0), (1, 'tts', 'b', 10),
                                          (0, 'tts', 'a', 20), (0, 'tts', 'b', 30),
                                          (0, 'speech', 'human', 40)):
            self.rows.append({'id': str(base), 'label': label, 'kind': kind, 'engine': engine,
                              'split': 'training', 'features_sha256': str(base), 'frames': 60,
                              'source_sha256': 'source-' + str(base), 'variant': 0,
                              '_feature_directory': 'head-features-tts-v1',
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

    def test_default_keeps_rows_without_optional_identity(self):
        without_ids = [{k: v for k, v in row.items() if k != 'id'} for row in self.rows]
        expected = Sampler(self.rows, 727)
        actual = Sampler(without_ids, 727)
        a, y = expected.batch()
        b, z = actual.batch()
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(y, z)

    def test_invalid_stride_and_heldout_mining_rejected(self):
        for value in (True, 0, 1, 2.5):
            with self.assertRaises(ValueError):
                Sampler(self.rows, 727, self.mined, hard_negative_stride=value)
        self.rows[-1]['split'] = 'validation'
        with self.assertRaises(ValueError):
            Sampler(self.rows, 727, self.mined)

    def endpoint_record(self, row):
        record = {key: row[key] for key in ('id', 'source_sha256', 'features_sha256', 'engine', 'variant')}
        record.update(feature_directory=row['_feature_directory'], old_keyword_end=row['keyword_end'],
                      new_keyword_end=2.08)
        return record

    def test_endpoint_sampling_keeps_negative_draws_rng_and_original_ptq(self):
        original = copy.deepcopy(self.rows)
        control = Sampler(self.rows, 727)
        treatment = Sampler(self.rows, 727, positive_endpoints=[self.endpoint_record(self.rows[0])])
        for _ in range(10):
            a, labels = control.batch()
            b, other_labels = treatment.batch()
            np.testing.assert_array_equal(labels, other_labels)
            negative = labels[:, 0] == 0
            np.testing.assert_array_equal(a[negative], b[negative])
            self.assertGreater(np.any(a != b, axis=(1, 2, 3)).sum(), 0)
            self.assertEqual(control.rng.bit_generator.state, treatment.rng.bit_generator.state)
        for a, b in zip(self.rows, original):
            self.assertEqual(a['keyword_end'], b['keyword_end'])
            np.testing.assert_array_equal(a['array'], b['array'])
        reference = Sampler(original, 728)
        ptq = Sampler(self.rows, 728)
        for _ in range(4):
            np.testing.assert_array_equal(reference.batch()[0], ptq.batch()[0])

    def test_endpoint_identity_scope_and_range_rejected(self):
        record = self.endpoint_record(self.rows[0])
        for change in ({'new_keyword_end': float('nan')}, {'new_keyword_end': True},
                       {'new_keyword_end': 1.0}, {'new_keyword_end': 2.8},
                       {'old_keyword_end': 3.0}, {'source_sha256': 'changed'},
                       {'features_sha256': 'changed'}, {'id': 'unknown'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                endpoint_map(self.rows, [{**record, **change}], 25)
        with self.assertRaises(ValueError):
            endpoint_map(self.rows, [record, record], 25)
        with self.assertRaises(ValueError):
            endpoint_map(self.rows, [self.endpoint_record(self.rows[2])], 25)
        self.rows[0]['split'] = 'validation'
        with self.assertRaises(ValueError):
            endpoint_map(self.rows, [record], 25)


if __name__ == '__main__':
    unittest.main()
