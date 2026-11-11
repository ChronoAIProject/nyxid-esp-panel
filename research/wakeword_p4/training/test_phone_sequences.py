"""CTC sequence packing and edit counts, independent of any model weights."""
import unittest

import numpy as np

from phone_sequences import collapse, edit_distance, pack_teacher_targets


class PhoneSequencesTest(unittest.TestCase):
    def test_repeats_separated_by_blank_remain_distinct(self):
        q = np.eye(4, dtype=np.float32)[[[1, 1, 0, 1, 2, 2], [0, 3, 0, 2, 2, 0]]]
        packed, sequences = pack_teacher_targets(q, 4)
        self.assertEqual(sequences, [[1, 1, 2], [3, 2]])
        np.testing.assert_array_equal(packed.argmax(-1), [[1, 1, 2, 0], [3, 2, 0, 0]])
        with self.assertRaises(ValueError):
            pack_teacher_targets(q, 3)

    def test_empty_and_invalid_teacher_targets_are_rejected(self):
        blank = np.eye(3, dtype=np.float32)[[[0, 0, 0]]]
        with self.assertRaises(ValueError):
            pack_teacher_targets(blank, 4)
        for value in (float('nan'), -.1, .5):
            bad = blank.copy(); bad[0, 0, 1] = value
            with self.assertRaises(ValueError):
                pack_teacher_targets(bad, 4)

    def test_decode_and_edit_distance_keep_insertions_deletions_and_repeats(self):
        self.assertEqual(collapse([1, 1, 0, 1, 2, 2, 0]), [1, 1, 2])
        self.assertEqual(edit_distance([1, 1, 2], [1, 2]), 1)
        self.assertEqual(edit_distance([1, 2], [1, 3, 2]), 1)
        self.assertEqual(edit_distance([1, 2], [2, 1]), 2)
        self.assertEqual(edit_distance([], [1, 2]), 2)


if __name__ == '__main__':
    unittest.main()
