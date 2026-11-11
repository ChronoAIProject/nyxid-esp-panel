import json
import unittest

import numpy as np

from old_source_replacement import ENGINES, SLOTS, OldSourceReplacement, build_groups, sampling_tables


def fixture():
    sources, records, labels = [], [], []
    for engine in ENGINES:
        for phrase, label in [('hey nyx', 1), ('hey nick', 0), ('hey next', 0), ('hey mix', 0)]:
            for voice in ['voice-a', 'voice-b']:
                index = len(sources);digest = f'hash-{index}'
                alias = {'phrase': phrase, 'label': label, 'engine': engine, 'sha256': digest, 'speakers': [voice]}
                sources.append({'source_sha256': digest, 'aliases': [alias]})
                for variant in range(3):
                    records.append({'source_index': index, 'pack_index': len(records),
                                    'source_sha256': digest, 'variant': variant})
                    labels.append(label)
    return sources, records, np.asarray(labels, dtype=np.float32)


def worker(seed=137726):
    sources, records, labels = fixture()
    value = OldSourceReplacement.__new__(OldSourceReplacement)
    value.groups = build_groups(sources, records, labels)
    value.tables = sampling_tables(value.groups)
    value.labels = labels
    value.features = np.arange(len(labels), dtype=np.uint16)[:, None, None]*np.ones((1, 150, 40), dtype=np.uint16)
    value.rng = np.random.default_rng(seed)
    return value


class OldSourceReplacementTest(unittest.TestCase):
    def test_alias_rows_do_not_add_source_mass(self):
        sources, records, labels = fixture()
        before = build_groups(sources, records, labels)
        sources[0]['aliases'].append(dict(sources[0]['aliases'][0]))
        self.assertEqual(build_groups(sources, records, labels), before)
        sources[0]['aliases'].append({**sources[0]['aliases'][0], 'label': 0})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            build_groups(sources, records, labels)

    def test_exact_replaced_slots_and_untouched_inputs_labels_weights(self):
        value = worker()
        x = np.random.default_rng(137727).uniform(0, 26, (128, 150, 40)).astype(np.float32)
        y = np.asarray([1]*32+[0]*96, dtype=np.float32)[:, None]
        weights = np.where(y[:, 0] > 0, 1, 20).astype(np.float32)
        original = [a.copy() for a in (x, y, weights)]
        result = value.replace(x, y, weights)
        kept = sorted(set(range(128))-set(SLOTS))
        self.assertEqual(len(kept), 88)
        np.testing.assert_array_equal(result[kept], x[kept])
        np.testing.assert_array_equal(result[list(SLOTS)], np.clip(value.features[value.last_indices].astype(np.float32)/25.6, 0, 26))
        np.testing.assert_array_equal(value.labels[value.last_indices], [1]*16+[0]*24)
        for actual, saved in zip((x, y, weights), original, strict=True):
            np.testing.assert_array_equal(actual, saved)

    def test_deterministic_independent_generator(self):
        first, second = worker(), worker()
        legacy = np.random.get_state()
        for _ in range(20):
            np.testing.assert_array_equal(first.sample_indices(), second.sample_indices())
        after = np.random.get_state()
        self.assertEqual(legacy[0], after[0])
        np.testing.assert_array_equal(legacy[1], after[1])
        self.assertEqual(legacy[2:], after[2:])

    def test_invalid_baseline_fails_before_draw(self):
        value = worker()
        x = np.zeros((128, 150, 40), dtype=np.float32)
        y = np.zeros((128, 1), dtype=np.float32)
        before = json.dumps(value.rng.bit_generator.state)
        with self.assertRaisesRegex(ValueError, 'phrase layout'):
            value.replace(x, y, np.ones(128))
        self.assertEqual(before, json.dumps(value.rng.bit_generator.state))


if __name__ == '__main__':
    unittest.main()
