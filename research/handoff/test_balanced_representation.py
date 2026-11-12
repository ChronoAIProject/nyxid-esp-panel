"""Numerical and cohort invariants for the all-source float diagnostic."""
import itertools
import unittest

import numpy as np

from balanced_representation import (
    ARMS, PHRASES, SPEEDS, across_ids, common_ends, group_measurements,
    grouped_records, paired_metrics,
)


class BalancedRepresentationTests(unittest.TestCase):
    def test_shared_time_does_not_shift_each_arm_to_its_own_tail(self):
        dry = dict(feature_crop_end=256, raw_shape=[355, 40])
        rir = dict(feature_crop_end=262, raw_shape=[361, 40])
        np.testing.assert_array_equal(common_ends([dry, rir]), np.arange(264, 351, 3))
        # A final full window needs the unused phase1 suffix to exist.
        short = dict(feature_crop_end=256, raw_shape=[270, 40])
        np.testing.assert_array_equal(common_ends([dry, rir, short]), [264, 267])
        with self.assertRaises(ValueError):
            common_ends([dict(feature_crop_end=280, raw_shape=[270, 40])])

    def test_complete_census_is_order_independent_and_missing_cells_fail(self):
        records = []
        for voice, speed, phrase, arm in itertools.product(["9", "2"], SPEEDS, PHRASES, ARMS):
            records.append(dict(voice=voice, length_multiplier=speed, target=phrase,
                                arm=arm, split="training", mechanical_pass=True,
                                entire_signal_fits=True, feature_crop_end=256,
                                raw_shape=[355, 40]))
        groups = grouped_records(list(reversed(records)), ["9", "2"])
        self.assertEqual([(v, s) for v, s, _, _ in groups],
                         list(itertools.product(["2", "9"], SPEEDS)))
        self.assertEqual([(r["target"], r["arm"]) for r in groups[0][2]],
                         list(itertools.product(PHRASES, ARMS)))
        for invalid in (records[:-1], records + [records[0]],
                        [{**records[0], "split": "validation"}] + records[1:]):
            with self.assertRaises(ValueError):
                grouped_records(invalid, ["9", "2"])

    def test_known_parallel_orthogonal_and_undefined_cosine(self):
        left = np.array([[1, 0], [0, 0], [1, 0]], dtype=float)
        right = np.array([[1, 2], [0, 0], [-1, 0]], dtype=float)
        result = paired_metrics(left, right, [3, 0])
        np.testing.assert_array_equal(result["l2"], [2, 0, 2])
        np.testing.assert_array_equal(result["logit_delta"], [0, 0, -6])
        np.testing.assert_array_equal(result["parallel_l2"], [0, 0, 2])
        np.testing.assert_array_equal(result["orthogonal_l2"], [2, 0, 0])
        self.assertAlmostEqual(result["cosine_distance"][0], 1 - 1 / np.sqrt(5))
        self.assertTrue(np.isnan(result["cosine_distance"][1]))
        self.assertEqual(result["cosine_distance"][2], 2)
        zero_head = paired_metrics(left, right, [0, 0])
        self.assertTrue(np.isnan(zero_head["parallel_l2"]).all())
        self.assertTrue(np.isnan(zero_head["orthogonal_l2"]).all())

    def test_actual_classifier_projection_and_all_comparisons_retained(self):
        z = np.zeros((4, 5, 2, 300))
        z[..., 0] = np.arange(20).reshape(4, 5, 1)
        weight = np.zeros(300)
        weight[0] = 1
        probabilities = 1 / (1 + np.exp(-(z[..., 0] - 10)))
        result = group_measurements(z, probabilities, weight, -10)
        self.assertEqual(len(result), 4 * 5 * 2 + (4 * 4 + 3 * 5) * 7)
        self.assertEqual(result["nuisance/hey_nyx/noise25/l2"]["mean"], 1)
        self.assertEqual(result["near_word/hey_nick/quiet/logit_delta"]["mean"], 5)
        self.assertEqual(result["nuisance/hey_nyx/noise25/cosine_distance"]["undefined"], 2)
        with self.assertRaises(ValueError):
            group_measurements(z, 1 - probabilities, weight, -10)

    def test_aggregate_equal_ids_and_explicit_incomplete_metrics(self):
        rows = []
        for voice, speeds in [("2", [1, 2, 3]), ("9", [10, 20, 30])]:
            for speed, value in zip(SPEEDS, speeds):
                # Unequal window counts cannot give the longer group more mass.
                rows.append(dict(voice=voice, speed=speed, measurements=dict(
                    known=dict(count=int(value), defined=int(value), undefined=0, mean=value),
                    zero=dict(count=2, defined=0, undefined=2, mean=None))))
        summary = across_ids(rows)
        self.assertEqual(summary["known"]["mean"], 11)
        self.assertEqual(summary["known"]["count"], 2)
        self.assertEqual(summary["zero"]["undefined_conditioning_ids"], 2)
        self.assertEqual(summary["zero"]["undefined_groups"], 6)
        self.assertEqual(summary["zero"]["undefined_windows"], 12)
        self.assertIsNone(summary["zero"]["mean"])
        with self.assertRaises(ValueError):
            across_ids(rows[:-1])


if __name__ == "__main__":
    unittest.main()
