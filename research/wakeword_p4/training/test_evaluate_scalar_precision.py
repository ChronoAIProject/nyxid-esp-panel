"""Independent precision-error contingency and unchanged strict-gate checks."""
import unittest

from evaluate_scalar_precision import paired_counts, strict_point


class PrecisionAccountingTest(unittest.TestCase):
    def test_quantization_can_both_add_and_remove_detections(self):
        values = [(1, .9, .9), (1, .9, .1), (1, .1, .9), (1, .1, .1),
                  (0, .9, .9), (0, .1, .9), (0, .1, .1)]
        rows = [{'label': label, 'arms': {'float': {'max_score': f}, 'int8': {'max_score': q}}}
                for label, f, q in values]
        result = paired_counts(rows, .5)
        self.assertEqual(result['positive'], {'views': 4, 'both_above': 1, 'float_only_above': 1,
                                             'int8_only_above': 1, 'neither_above': 1})
        self.assertEqual(result['negative'], {'views': 3, 'both_above': 1, 'float_only_above': 0,
                                             'int8_only_above': 1, 'neither_above': 1})

    def test_zero_event_and_both_group_gates_are_required(self):
        def point(t, recall, events=0, source=.01, corpus=.01):
            return {'threshold': t, 'macro_recall': recall, 'ambient_false_events': events,
                    'max_source_false_rate': source, 'max_corpus_false_rate': corpus}
        good = point(.5, .8)
        self.assertEqual(strict_point([good, point(.6, .9, events=1),
                                       point(.7, .95, source=.02), point(.8, .99, corpus=.02)]), good)
        self.assertIsNone(strict_point([point(.5, .99, events=1)]))
        self.assertEqual(strict_point([good, point(.6, .8)])['threshold'], .5)


if __name__ == '__main__':
    unittest.main()
