"""Regression scenarios for always-listening wake event accounting."""
import unittest

from evaluate import event_indices, events


class WakeEvents(unittest.TestCase):
    def test_premature_activation_suppresses_later_keyword_peak(self):
        scores = [0.] * 5 + [.99] * 80 + [0.] * 80 + [.99] * 5
        # A second threshold crossing while still latched is not a new wake.
        self.assertEqual(event_indices(scores, .97), [5, 165])
        self.assertEqual(events(scores, .97), 2)

    def test_short_silence_does_not_bypass_refractory(self):
        scores = [1.] + [0.] * 10 + [1.] * 10 + [0.] * 60 + [1.]
        self.assertEqual(event_indices(scores, .97), [0, 81])

    def test_continuously_high_score_is_one_false_event(self):
        self.assertEqual(events([1.] * 1000, .97), 1)
        self.assertEqual(events([0.] * 1000, .97), 0)


if __name__ == "__main__":
    unittest.main()
