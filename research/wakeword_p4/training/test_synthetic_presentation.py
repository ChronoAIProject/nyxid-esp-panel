"""Critical presentation-window and candidate-promotion gate boundaries."""
import copy
import unittest
import numpy as np
from calibrate_slr83_development import passes
from synthetic_presentation import metrics


def clip(target='hey_nyx', pulse=None, lead=2720, end=4160, native_end=3200):
    scores = np.zeros(100)
    if pulse is not None:
        scores[pulse] = 1
    row = dict(path='fixture.wav', voice='fixture', variant=0, target=target,
               intended_phrase=target, lead_samples=lead, keyword_end=end,
               native_generation_end=native_end, samples=64000)
    return row, scores, []


class PresentationTests(unittest.TestCase):
    def test_generation_and_rir_references_are_distinct(self):
        m = metrics([clip(pulse=0)], .5)
        self.assertEqual(m['presentation_trigger_clips'], 1)
        r = m['records'][0]
        self.assertAlmostEqual(r['first_trigger_relative_generation_and_rir_end_ms'], -90)
        self.assertAlmostEqual(r['first_trigger_relative_native_generation_end_ms'], -30)

    def test_window_includes_start_and_end_but_excludes_late(self):
        self.assertEqual(metrics([clip(pulse=0)], .5)['presentation_trigger_clips'], 1)
        # index28 is1.01s, exactly generation .21s plus .8s.
        self.assertEqual(metrics([clip(pulse=28, end=3360)], .5)['presentation_trigger_clips'], 1)
        self.assertEqual(metrics([clip(pulse=29, end=3360)], .5)['presentation_trigger_clips'], 0)

    def test_pre_generation_events_cannot_satisfy_gate(self):
        m = metrics([clip(pulse=0, lead=3200)], .5)
        self.assertEqual(m['presentation_trigger_clips'], 0)
        self.assertEqual(m['pre_generation_positive_events'], 1)
        self.assertFalse(passes(m, .1, 0))

    def test_each_confusing_or_cross_phrase_blocks_gate(self):
        for target in (None, 'hi_esp'):
            m = metrics([clip(pulse=0), clip(target=target, pulse=0)], .5)
            self.assertFalse(passes(m, .1, .5))

    def test_far_and_baseline_macro_are_both_required(self):
        m = metrics([clip(pulse=0), clip(pulse=None)], .5)
        self.assertTrue(passes(m, .5, .5))
        self.assertFalse(passes(m, .500001, .5))
        self.assertFalse(passes(m, .5, .500001))
        bad = copy.deepcopy(m)
        bad['pre_generation_positive_events'] = 1
        self.assertFalse(passes(bad, .5, .5))


if __name__ == '__main__':
    unittest.main()
