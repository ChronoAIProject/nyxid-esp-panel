"""Catch cached-score accounting changes without replaying a model."""
import copy
import unittest

import numpy as np

from audit_scalar_wake_scores import equal, reconstruct_clip


class ScoreAuditTest(unittest.TestCase):
    def test_early_prefix_peak_cannot_replace_keyword_window_maximum(self):
        row = {'path': 'source.wav', 'variant': 0, 'label': 1, 'engine': 'test',
               'text': 'Hey Nyx.', 'frames': 60, 'keyword_frame': 30}
        times = (3*(np.arange(16)+4)+5)*.01
        scores = np.zeros(16); scores[0], scores[4] = 1., .8
        record = {'source': row['path'], **{k: row[k] for k in ('variant', 'label', 'engine', 'text')},
                  'times': times.tolist(), 'scores': scores.tolist(),
                  'keyword_end': .32, 'max_score': .8}
        clip, count = reconstruct_clip(row, record)
        self.assertEqual(count, 16)
        self.assertEqual(clip['max_score'], .8)
        for field, bad in (('max_score', 1.), ('times', (times+.01).tolist()),
                           ('scores', [float('nan')]*16), ('source', 'other.wav')):
            changed = copy.deepcopy(record); changed[field] = bad
            with self.assertRaises(ValueError):
                reconstruct_clip(row, changed)

    def test_roundoff_allowance_cannot_hide_counts_or_structure(self):
        self.assertTrue(equal({'recall': .1+.2, 'events': 1}, {'recall': .3, 'events': 1}))
        self.assertFalse(equal({'events': 1}, {'events': 2}))
        self.assertFalse(equal({'events': 1}, {'events': 1.}))
        self.assertFalse(equal({'recall': .3}, {'recall': .300000001}))
        self.assertFalse(equal({'recall': float('nan')}, {'recall': float('nan')}))
        self.assertFalse(equal([1, 2], [1]))


if __name__ == '__main__':
    unittest.main()
