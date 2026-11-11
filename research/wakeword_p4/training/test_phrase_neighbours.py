"""Numerical DTW checks and training-only source-selection boundaries."""
import unittest

import numpy as np

from audit_phrase_neighbours import dtw_distance, source_bounds, training_rows


class NeighboursTest(unittest.TestCase):
    def test_exhaustive_paths_and_symmetry(self):
        left = np.array([[1., 3, 2], [4, 0, 3]])
        right = np.array([[0., 1, 3], [3, 0, 4], [4, 2, 1]])
        def normalized(a):
            a = a-a.mean(axis=1, keepdims=True)
            return a/np.linalg.norm(a, axis=1, keepdims=True)
        cost = np.clip(1-normalized(left)@normalized(right).T, 0, 2)
        paths = []
        def walk(i, j, total):
            total += cost[i, j]
            if (i, j) == (1, 2):
                paths.append(total); return
            for di, dj in ((0, 1), (1, 0), (1, 1)):
                if i+di < 2 and j+dj < 3:
                    walk(i+di, j+dj, total)
        walk(0, 0, 0.)
        self.assertAlmostEqual(dtw_distance(left, right), min(paths)*2/5, places=14)
        self.assertAlmostEqual(dtw_distance(left, right), dtw_distance(right, left), places=14)

    def test_repeated_frames_gain_offset_and_degenerate_inputs(self):
        value = np.array([[1., 2, 4], [4, 3, 0], [0, 1, 4]])
        unchanged = value.copy()
        self.assertAlmostEqual(dtw_distance(value, np.repeat(value*2+5, 2, axis=0)), 0., places=14)
        np.testing.assert_array_equal(value, unchanged)
        self.assertEqual(dtw_distance(np.zeros((2, 3)), np.ones((3, 3))), 0.)
        for invalid in (np.empty((0, 3)), np.array([[np.nan, 1, 2]]), np.zeros(3)):
            with self.assertRaises(ValueError):
                dtw_distance(value, invalid)
        pcm = np.r_[np.zeros(160), np.full(320, .1), np.zeros(160)]
        self.assertEqual(source_bounds(pcm), (1, 3))
        with self.assertRaises(ValueError):
            source_bounds(np.zeros(320))

    def test_reserved_source_or_bank_conflict_is_rejected(self):
        row = {'path': 'training/a.wav', 'variant': 0, 'split': 'training',
            'engine': 'piper', 'sha256': 'a'*64, 'label': 1, 'text': 'Hey Nyx.', 'speakers': ['x']}
        bank = {'rows': [{'path': row['path'], 'variant': 0, 'source_sha256': row['sha256'], 'label': 1}]}
        self.assertEqual(training_rows([row], bank)[0]['class'], 0)
        for index, changed_bank in (([row, {**row, 'path': 'reserved/a.wav', 'split': 'testing'}], bank),
                ([{**row, 'split': 'validation'}], bank),
                ([row], {'rows': [{**bank['rows'][0], 'label': 0}]})):
            with self.assertRaises(ValueError):
                training_rows(index, changed_bank)


if __name__ == '__main__':
    unittest.main()
