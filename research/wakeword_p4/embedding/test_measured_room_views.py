"""Protect response timing, level matching and paired random draws."""
import unittest

import numpy as np

import prepare_measured_room_views as measured


class MeasuredRoomTests(unittest.TestCase):
    def tearDown(self):
        measured.ROOM = None

    def test_leading_delay_removed_and_energy_matched(self):
        original = np.array([0, 0, .01, 1, .5, -.25], np.float64)
        response, onset = measured.align_energy(original, 4.)
        np.testing.assert_array_equal(original, [0, 0, .01, 1, .5, -.25])
        self.assertEqual(onset, 3)
        self.assertEqual(len(response), 3)
        self.assertAlmostEqual(float(np.sum(response.astype(np.float64) ** 2)), 4., places=6)
        self.assertGreater(response[0], 0)

    def test_reference_random_sequence_and_energy_preserved(self):
        left, right = np.random.default_rng(733), np.random.default_rng(733)
        expected = measured.ORIGINAL_ROOM_RESPONSE(left)
        measured.ROOM = {'pcm': np.array([0., .5, -.25, .125]), 'sha256': 'fixture'}
        actual = measured.matched_room_response(right)
        np.testing.assert_array_equal(left.random(30), right.random(30))
        self.assertAlmostEqual(float(np.sum(actual.astype(np.float64) ** 2)),
                               float(np.sum(expected.astype(np.float64) ** 2)), places=6)
        self.assertEqual(measured.ROOM['response_metadata']['rir_onset_samples'], 1)

    def test_invalid_room_inputs_rejected(self):
        for pcm, energy in [([], 1), ([0, 0], 1), ([np.nan, 1], 1),
                            ([[1, 2]], 1), ([1, 2], 0), ([1, 2], np.inf)]:
            with self.subTest(pcm=pcm, energy=energy), self.assertRaises(ValueError):
                measured.align_energy(pcm, energy)


if __name__ == '__main__':
    unittest.main()
