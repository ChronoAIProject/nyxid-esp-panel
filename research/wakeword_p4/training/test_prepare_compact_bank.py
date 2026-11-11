"""Independent pairing/provenance checks for the uncapped compact bank."""
import copy
import unittest

from prepare_compact_bank import coverage, select_pairs


def fixture():
    rows, bounds = [], {}
    for speaker in ('a', 'b'):
        for variant in (0, 1):
            for number, text in enumerate(('Hey Nyx.', 'Hey Nix.', 'Hey Mix.', 'Hey Next.', 'Hey Nick.')):
                path = f'fixture/{speaker}/{number}.wav'
                row = {'path': path, 'sha256': f'{speaker}-{number}', 'variant': variant,
                       'text': text, 'label': int(number < 2), 'split': 'training',
                       'engine': 'fixture', 'speakers': [speaker], 'frames': 390,
                       'lead_frame': 40, 'keyword_frame': 340}
                rows.append(row)
                if not row['label']:
                    bounds[path, variant] = (345, 365)
    return rows, bounds


class BroadBankTest(unittest.TestCase):
    def test_full_cartesian_coverage_and_invariant_order(self):
        rows, bounds = fixture()
        pairs = select_pairs(rows, bounds)
        self.assertEqual(pairs, select_pairs(list(reversed(rows)), bounds))
        measured = coverage(pairs)
        self.assertEqual(measured['pairs'], 24)
        self.assertEqual(measured['rows'], 48)
        self.assertEqual(measured['unique_crops'], 20)
        self.assertEqual(measured['unique_sources'], 10)
        self.assertEqual(measured['positive_sources'], 4)
        self.assertEqual(measured['negative_sources'], 6)
        seen = set()
        for stratum, ((p, pb), (n, nb)) in pairs:
            self.assertEqual(p['label'], 1)
            self.assertEqual(n['label'], 0)
            self.assertEqual(p['variant'], n['variant'])
            self.assertEqual(p['speakers'], n['speakers'])
            self.assertTrue(335 <= sum(pb)//2 <= p['frames'])
            self.assertEqual(nb, (345, 365))
            key = (p['path'], n['path'], p['variant'])
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_reserved_duplicate_ambiguous_or_unbound_sources_are_rejected(self):
        rows, bounds = fixture()
        held = copy.deepcopy(rows[0]); held.update(path='reserved.wav', split='validation')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            select_pairs(rows+[held], bounds)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            select_pairs(rows+[copy.deepcopy(rows[0])], bounds)
        changed = copy.deepcopy(rows)
        for row in changed:
            if not row['label']:
                row['speakers'] = ['unmatched']
        with self.assertRaisesRegex(ValueError, 'all confusers'):
            select_pairs(changed, bounds)
        with self.assertRaisesRegex(ValueError, 'all confusers'):
            select_pairs(rows, {})


if __name__ == '__main__':
    unittest.main()
