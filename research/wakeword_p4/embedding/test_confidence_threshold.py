"""A zero observed rate is insufficient when exposure is short."""
import math
import unittest

from select_confidence_threshold import select, upper95


class ConfidenceThresholdTests(unittest.TestCase):
    def row(self, hours, threshold=.4, recall=.95):
        return {'continuous_events': 0, 'continuous_hours': hours,
                'continuous_events_per_hour_upper_95': upper95(0, hours),
                'macro_recall': recall, 'threshold': threshold, 'max_corpus_false_rate': .1}

    def test_zero_count_exposure_and_no_fallback(self):
        self.assertAlmostEqual(upper95(0, 6), -math.log(.05) / 6)
        self.assertIsNone(select([self.row(1)], .5))
        self.assertEqual(select([self.row(6)], .5)['threshold'], .4)

    def test_no_manufactured_bound_or_invalid_exposure(self):
        bad = self.row(1)
        bad['continuous_events_per_hour_upper_95'] = 0
        with self.assertRaises(ValueError):
            select([bad], .5)
        for count, hours in [(-1, 1), (0, 0), (0, float('nan'))]:
            with self.assertRaises(ValueError):
                upper95(count, hours)

    def test_best_recall_within_bound(self):
        rows = [self.row(8, .4, .94), self.row(8, .6, .92), self.row(1, .2, .99)]
        self.assertEqual(select(rows, .5)['threshold'], .4)


if __name__ == '__main__':
    unittest.main()
