import unittest

from lexical_rms_alignment import aligned_record


class LexicalRmsAlignmentTest(unittest.TestCase):
    def test_standalone_aliases_share_source_energy_anchor(self):
        record = {'class': 0, 'minimum_end': 35, 'maximum_end': 38, 'id': 'original'}
        aligned = aligned_record(record, ['Hey Nyx.', 'Hey Nicks!'], .8, 60)
        self.assertEqual(aligned, record | {'minimum_end': 41, 'maximum_end': 44})
        self.assertEqual(record['minimum_end'], 35)

    def test_command_and_negative_records_keep_original_fields(self):
        record = {'class': 0, 'minimum_end': 35, 'maximum_end': 38}
        self.assertEqual(aligned_record(record, ['Hey Nyx, turn on the light.'], None, 60), record)
        self.assertEqual(aligned_record(record, ['Hey Nyx.', 'Hey Nyx, turn on the light.'], None, 60), record)
        negative = record | {'class': 1}
        self.assertEqual(aligned_record(negative, ['Hey Nick.'], None, 60), negative)

    def test_out_of_bounds_energy_anchor_rejected(self):
        record = {'class': 0, 'minimum_end': 35, 'maximum_end': 38}
        with self.assertRaisesRegex(ValueError, 'warmed'):
            aligned_record(record, ['Hey Nyx.'], 20., 60)


if __name__ == '__main__':
    unittest.main()
