"""Check macro/worst-corpus and continuous event accounting independently."""
import unittest

import numpy as np

from evaluate import events
from evaluate_compact_sequence import summarize


class SequenceEvaluationTest(unittest.TestCase):
    def test_macro_and_worst_corpus_do_not_get_diluted_by_other_views(self):
        rows = []
        for engine, label, source, values in (
                ('a', 1, 'p', [.9, .1]), ('b', 1, 'p', [.9]),
                ('a', 0, 'first', [.7, .1]), ('a', 0, 'second', [.9]),
                ('b', 0, 'other', [.1, .1, .1])):
            rows.extend({'engine': engine, 'label': label, 'source': source+'/file', 'max_score': v}
                        for v in values)
        ambient = [{'hours': 2., 'events': {'0.5': 2}},
                   {'hours': 1., 'events': {'0.5': 1}}]
        result = summarize(rows, ambient, .5)
        self.assertAlmostEqual(result['macro_recall'], .75)
        self.assertAlmostEqual(result['recall'], 2/3)
        self.assertAlmostEqual(result['hard_negative_false_rate'], 2/6)
        self.assertAlmostEqual(result['max_source_false_rate'], 2/3)
        self.assertEqual(result['max_corpus_false_rate'], 1.)
        self.assertEqual(result['ambient_hours'], 3.)
        self.assertEqual(result['ambient_false_events'], 3)
        self.assertGreater(result['ambient_false_per_hour_upper_95'], 1.)

    def test_pointwise_reduction_does_not_imply_fewer_latched_events(self):
        binary = np.full(200, .8)
        product = binary.copy()
        product[1:80] = 0.
        self.assertTrue(np.all(product <= binary))
        self.assertEqual(events(binary, .5), 1)
        self.assertEqual(events(product, .5), 2)


if __name__ == '__main__':
    unittest.main()
