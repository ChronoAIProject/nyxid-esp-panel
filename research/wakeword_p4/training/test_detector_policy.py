"""Prevent scoring one detector policy while deploying another."""
import unittest
import numpy as np

from detector_policy import calibrated_smoothing_steps, smooth


class DetectorPolicy(unittest.TestCase):
    def test_old_calibration_and_default_preserve_exact_historical_average(self):
        rng = np.random.default_rng(726)
        scores = rng.random(400).astype(np.float32)
        self.assertEqual(calibrated_smoothing_steps({}), 5)
        np.testing.assert_array_equal(smooth(scores), np.convolve(scores, np.ones(5)/5, mode="valid"))

    def test_bound_policy_changes_brief_peak_and_handles_short_input(self):
        scores = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        self.assertEqual(smooth(scores, 3).max(), 1)
        self.assertLess(smooth(scores, 7).max(), .5)
        for steps in (3, 5, 7):
            self.assertEqual(calibrated_smoothing_steps({"smoothing_steps": steps}), steps)
            self.assertEqual(smooth(scores[:steps-1], steps).size, 0)
            self.assertEqual(smooth(scores[:steps], steps).size, 1)

    def test_explicit_invalid_metadata_never_falls_back(self):
        for value in (None, 0, 1, 4, 8, 5., "5", True, False, float("nan"), [5]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    calibrated_smoothing_steps({"smoothing_steps": value})


if __name__ == "__main__":
    unittest.main()
