import unittest

import numpy as np
from negative_activity import interval_map, rms_span


class NegativeActivityTest(unittest.TestCase):
    def fixture(self):
        row = {'id': 'n', 'split': 'training', 'kind': 'tts', 'label': 0, 'engine': 'example',
               'source_sha256': 'source', 'features_sha256': 'feature', 'variant': 0, 'frames': 60,
               '_feature_directory': 'head-features-tts-v1'}
        record = {key: row[key] for key in ('id', 'source_sha256', 'features_sha256', 'engine', 'variant', 'frames')}
        record.update(feature_directory=row['_feature_directory'], minimum_end=32, maximum_end=42)
        return row, record

    def test_energy_span_preserves_lead_tail_and_internal_gap(self):
        pcm = np.r_[np.zeros(320), np.full(160, 100), np.zeros(320), np.full(160, 200), np.zeros(960)].astype(np.int16)
        original = pcm.copy()
        self.assertEqual(rms_span(pcm), (.02, .06))
        np.testing.assert_array_equal(pcm, original)
        with self.assertRaisesRegex(ValueError, 'Silent'):
            rms_span(np.zeros(100, np.int16))

    def test_map_binds_roles_identity_and_complete_coverage(self):
        row, record = self.fixture()
        self.assertEqual(interval_map([row], [record], 25), {'n': (32, 42)})
        for change in ({'source_sha256': 'other'}, {'features_sha256': 'other'},
                       {'minimum_end': True}, {'minimum_end': 24}, {'maximum_end': 61},
                       {'feature_directory': 'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                interval_map([row], [record | change], 25)
        for change in ({'split': 'validation'}, {'label': 1}, {'kind': 'speech'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                interval_map([row | change], [record], 25)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            interval_map([row], [record, record], 25)
        with self.assertRaisesRegex(ValueError, 'cover every'):
            interval_map([row, row | {'id': 'missing'}], [record], 25)


if __name__ == '__main__':
    unittest.main()
