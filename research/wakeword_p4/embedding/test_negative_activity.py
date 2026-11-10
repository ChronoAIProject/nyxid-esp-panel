import unittest

import numpy as np
from negative_activity import interval_map, rms_span
from negative_activity_fallback import uniform_fallback_ids
from audit_negative_activity import energy_statistics


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

    def test_nonzero_below_floor_is_distinguished_from_all_zero(self):
        pcm = np.zeros(4800, np.int16)
        pcm[300] = 1
        stats = energy_statistics(pcm)
        self.assertEqual(stats['active_blocks'], 0)
        self.assertEqual(stats['nonzero_samples'], 1)
        self.assertEqual(stats['sample_peak_absolute'], 1)
        self.assertLess(stats['peak_10ms_rms'], 1)
        self.assertIsNone(stats['activity_start_seconds'])
        self.assertEqual(energy_statistics(np.zeros(4800, np.int16))['nonzero_samples'], 0)

    def test_fallback_requires_explicit_full_original_range(self):
        row, record = self.fixture()
        fallback = record | {'minimum_end': 25, 'maximum_end': 60, 'activity_defined': False,
                             'sampling_policy': 'retain_original_uniform_endpoint'}
        self.assertEqual(uniform_fallback_ids([row], [fallback], 25), {'n'})
        self.assertEqual(uniform_fallback_ids([row], [record], 25), set())
        for change in ({'minimum_end': 26}, {'maximum_end': 59}, {'activity_defined': True},
                       {'sampling_policy': 'filter_source'}, {'sampling_policy': 'source_activity'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                uniform_fallback_ids([row], [fallback | change], 25)


if __name__ == '__main__':
    unittest.main()
