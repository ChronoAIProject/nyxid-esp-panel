import unittest
import numpy as np

from audit_head_events import trace_events
from evaluate_head import events


class HeadEventTraceTest(unittest.TestCase):
    def test_high_plateau_emits_once_and_needs_low_score_after_refractory(self):
        self.assertEqual(trace_events([.9] * 100, .8), [0])
        self.assertEqual(trace_events([.9] * 25 + [0, .9], .8), [0, 26])
        # A low sample before the refractory interval expires cannot rearm.
        self.assertEqual(trace_events([.9, 0] + [.9] * 98, .8), [0])

    def test_trace_count_reproduces_original_counter_on_fixed_and_random_streams(self):
        rng = np.random.default_rng(742)
        for threshold in [.05, .5, .98, .999]:
            for values in [[], [0] * 100, [1] * 100, rng.random(5000)]:
                self.assertEqual(len(trace_events(values, threshold)), events(values, threshold))

    def test_invalid_scores_are_rejected(self):
        for values in [[float('nan')], [-.01], [1.01], [[.5]]]:
            with self.assertRaises(ValueError):
                trace_events(values, .8)


if __name__ == '__main__':
    unittest.main()
