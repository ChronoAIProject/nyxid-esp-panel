"""Check hierarchical source weights, held-fold isolation and malformed labels."""
import copy
import unittest

import numpy as np

from intent_sampling import hierarchical_probabilities, distribution


def fixture():
    rows = []
    for engine in ('a', 'b'):
        for label in range(4):
            for source in range(1+label):
                for variant in range(1+source):
                    for crop in range(1+variant):
                        rows.append({'engine': engine, 'class': label,
                            'sha256': f'{1+1000*(engine == "b")+100*label+source:064x}',
                            'variant': variant, 'crop': crop, 'split': 'training'})
    return rows


class SamplingTest(unittest.TestCase):
    def test_exact_hierarchy_masses_ignore_unequal_crop_and_render_counts(self):
        rows = fixture(); ids = np.arange(len(rows))
        p = hierarchical_probabilities(rows, ids)
        report = distribution(rows, ids, p)
        for value in report.values():
            self.assertAlmostEqual(value['probability_mass'], 1/8, places=14)
            self.assertAlmostEqual(value['effective_source_hashes'], value['source_hashes'], places=12)
        for source in {r['sha256'] for r in rows}:
            positions = [i for i, r in enumerate(rows) if r['sha256'] == source]
            row = rows[positions[0]]
            self.assertAlmostEqual(p[positions].sum(), 1/(8*(1+row['class'])), places=14)
            for variant in {rows[i]['variant'] for i in positions}:
                group = [i for i in positions if rows[i]['variant'] == variant]
                self.assertAlmostEqual(p[group].sum(), p[positions].sum()/len({rows[i]['variant'] for i in positions}), places=14)

    def test_fold_uses_only_selected_training_rows_and_replays_draws(self):
        rows = fixture(); ids = np.array([i for i, r in enumerate(rows) if r['engine'] == 'a'])
        first = hierarchical_probabilities(rows, ids)
        altered = copy.deepcopy(rows)
        for row in altered:
            if row['engine'] == 'b':
                row.clear()  # Held-out content must not influence the hierarchy.
        np.testing.assert_array_equal(first, hierarchical_probabilities(altered, ids))
        left = np.random.default_rng(1223).choice(ids, 100, replace=True, p=first)
        right = np.random.default_rng(1223).choice(ids, 100, replace=True, p=first)
        np.testing.assert_array_equal(left, right)
        self.assertTrue(set(left) <= set(ids))
        # Explicit p=None retains the previous control's exact RNG path.
        np.testing.assert_array_equal(np.random.default_rng(1223).choice(ids, 100, replace=True),
            np.random.default_rng(1223).choice(ids, 100, replace=True, p=None))

    def test_rejects_invalid_fold_identity_and_missing_classes(self):
        rows = fixture(); ids = np.arange(len(rows))
        for bad in ([], [0, 0], [-1], [len(rows)], [0.5], [[0]]):
            with self.assertRaises(ValueError):
                hierarchical_probabilities(rows, bad)
        with self.assertRaises(ValueError):
            hierarchical_probabilities(rows, ids[[r['class'] != 3 for r in rows]])
        for key, value in [('split', 'validation'), ('sha256', 'invalid'), ('variant', -1), ('class', True)]:
            altered = copy.deepcopy(rows); altered[0][key] = value
            with self.assertRaises(ValueError):
                hierarchical_probabilities(altered, ids)
        altered = copy.deepcopy(rows); altered[-1]['sha256'] = altered[0]['sha256']
        with self.assertRaises(ValueError):
            hierarchical_probabilities(altered, ids)


if __name__ == '__main__':
    unittest.main()
