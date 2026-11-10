import unittest
import numpy as np

from lexical_auxiliary import AuxiliarySampler, CLASSES, lexical_class, rms_end


class LexicalAuxiliaryTest(unittest.TestCase):
    def fixtures(self):
        rows, records = [], []
        for cls in range(len(CLASSES)):
            row = {'id': str(cls), 'source_sha256': str(cls), 'features_sha256': str(cls),
                'split': 'training', 'kind': 'tts', 'engine': 'a', 'label': int(cls == 0), 'variant': 0,
                '_feature_directory': 'head-features-tts-v1', 'frames': 40,
                'array': np.full((40, 96), cls, np.int8)}
            rows.append(row)
            records.append({k: row[k] for k in ('id', 'source_sha256', 'features_sha256', 'engine', 'label', 'variant')}
                           | {'class': cls, 'minimum_end': 25, 'maximum_end': 35})
        return rows, records

    def test_aliases_stay_positive_and_only_exact_negatives_classified(self):
        self.assertEqual(lexical_class(1, ['Hey Nyx.', 'Hey Nicks!']), 0)
        self.assertEqual(lexical_class(0, ['Hey, Next!']), 2)
        self.assertIsNone(lexical_class(0, ['Hey Next, what time is it?']))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            lexical_class(0, ['Hey Nick.', 'Hey Nice.'])

    def test_source_tail_does_not_move_auxiliary_rms_anchor(self):
        pcm = np.concatenate([np.zeros(160), np.full(320, 200), np.zeros(1600)])
        self.assertEqual(rms_end(pcm), .03)
        with self.assertRaisesRegex(ValueError, 'Silent'):
            rms_end(np.zeros(500, np.int16))

    def test_auxiliary_samples_keep_feature_bytes_labels_and_determinism(self):
        rows, records = self.fixtures()
        a, b = AuxiliarySampler(rows, records, 721), AuxiliarySampler(rows, records, 721)
        for _ in range(3):
            x, y = a.batch()
            xx, yy = b.batch()
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)
            np.testing.assert_array_equal(x[:, 0, 0, 0] * 128, y)

    def test_heldout_wrong_labels_duplicates_and_bounds_rejected(self):
        for field, value in [('split', 'validation'), ('label', 0), ('features_sha256', 'changed')]:
            rows, records = self.fixtures()
            rows[0][field] = value
            with self.assertRaises(ValueError):
                AuxiliarySampler(rows, records, 721)
        rows, records = self.fixtures()
        with self.assertRaises(ValueError):
            AuxiliarySampler(rows, records + [records[0]], 721)
        records[0]['maximum_end'] = 41
        with self.assertRaises(ValueError):
            AuxiliarySampler(rows, records, 721)


if __name__ == '__main__':
    unittest.main()
