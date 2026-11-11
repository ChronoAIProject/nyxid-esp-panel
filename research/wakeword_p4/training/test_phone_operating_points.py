"""Preserve old selection while allowing a predeclared dominated bad control."""
import unittest

from compare_phone_distillation import operating_points


def row(threshold, recall, rate, events):
    return {'threshold': threshold, 'macro_recall': recall, 'max_source_false_rate': rate,
            'max_corpus_false_rate': rate, 'ambient_false_events': events}


class OperatingPointsTest(unittest.TestCase):
    def test_old_no_zero_event_control_remains_unadvanced_without_opt_in(self):
        arms = {'control': {'grid': [row(.97, .8, .05, 1)]},
                'phonetic': {'grid': [row(.97, .9, .005, 0)]}}
        self.assertFalse(operating_points(arms)['advance'])
        chosen = operating_points(arms, .97)
        self.assertTrue(chosen['advance'])
        self.assertEqual(chosen['control_comparator_policy'], 'frozen_reference_with_false_rate_dominance')

    def test_fallback_preserves_both_absolute_and_control_false_rate_limits(self):
        arms = {'control': {'grid': [row(.97, .8, .002, 1)]},
                'phonetic': {'grid': [row(.95, .95, .009, 0), row(.97, .85, .001, 0)]}}
        chosen = operating_points(arms, .97)
        self.assertTrue(chosen['advance'])
        self.assertEqual(chosen['treatment_comparator']['threshold'], .97)
        arms['phonetic']['grid'] = [row(.95, .95, .009, 0)]
        self.assertFalse(operating_points(arms, .97)['advance'])
        arms['phonetic']['grid'] = [row(.97, .95, .001, 1)]
        self.assertFalse(operating_points(arms, .97)['advance'])

    def test_existing_viable_control_and_three_point_requirement_are_preserved(self):
        arms = {'control': {'grid': [row(.95, .8, .005, 0)]},
                'phonetic': {'grid': [row(.97, .82, .001, 0)]}}
        chosen = operating_points(arms, .97)
        self.assertFalse(chosen['advance'])
        self.assertEqual(chosen['control_comparator_policy'], 'strict_gate')
        arms['phonetic']['grid'][0]['macro_recall'] = .84
        self.assertTrue(operating_points(arms, .97)['advance'])


if __name__ == '__main__':
    unittest.main()
