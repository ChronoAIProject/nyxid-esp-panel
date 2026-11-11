"""Independent path-sum and label checks for the compact phrase diagnostic."""
import itertools
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from phone_sequences import collapse
from phrase_ctc_readout import hypothesis_scores, load_bank, phrase_class, summary, targets


class PhraseReadoutTest(unittest.TestCase):
    def test_likelihood_matches_exhaustive_paths_and_respects_order(self):
        logits = np.array([[[0, 3, -2, -3, -4, -5], [2, 1, 0, 0, 0, 0],
                            [0, -2, 4, 2, 1, 0], [1, -1, 2, 3, 0, 0]]], np.float32)
        probabilities = np.exp(logits[0]-logits[0].max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        totals = {tuple(target): 0. for target in targets(np.arange(4))}
        for path in itertools.product(range(6), repeat=4):
            target = tuple(collapse(path))
            if target in totals:
                totals[target] += np.prod([probabilities[i, token] for i, token in enumerate(path)])
        np.testing.assert_allclose(hypothesis_scores(logits)[0], np.log(list(totals.values())), rtol=1e-6, atol=1e-6)
        forward = np.full((1, 3, 6), -8., np.float32)
        forward[0, 0, 1] = forward[0, 1, 0] = forward[0, 2, 2] = 8
        self.assertGreater(hypothesis_scores(forward)[0, 0], hypothesis_scores(forward[:, ::-1])[0, 0]+10)

    def test_original_aliases_only_and_conflicting_labels_rejected(self):
        for text in ('Hey Nyx.', 'Hey Nix!', 'Hey Nicks?'):
            self.assertEqual(phrase_class({'text': text, 'label': 1}), 0)
        for text in ('Hey Next.', 'Hey Nick.', 'Hey Mix.'):
            self.assertNotEqual(phrase_class({'text': text, 'label': 0}), 0)
        for text, label in [('Hey Nyx.', 0), ('Hey Next.', 1), ('Hey Nyx, turn on the light.', 1)]:
            with self.assertRaises(ValueError):
                phrase_class({'text': text, 'label': label})
        for values in ([-1], [4], [1.5], [[0]]):
            with self.assertRaises(ValueError):
                targets(values)

    def test_binary_errors_preserve_every_confuser_and_per_class_counts(self):
        result = summary(np.array([0, 0, 1, 2, 3]), np.array([0, 2, 0, 2, 3]))
        self.assertEqual(result['nyx_recall'], .5)
        self.assertEqual(result['confuser_rows'], 3)
        self.assertEqual(result['confuser_false_nyx'], 1)
        self.assertAlmostEqual(result['confuser_false_nyx_fraction'], 1/3)
        self.assertAlmostEqual(result['macro_accuracy'], .625)

    def test_bank_rejects_changed_features_and_reserved_source_overlap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            x = np.stack([np.full((335, 40), i, np.uint16) for i in range(6)])
            x.astype('<u2').tofile(root/'features.bin')
            np.savez(root/'bank.npz', features=x)
            original, bank = [], []
            for i in range(6):
                label = 1-i % 2
                path = f'{i}.wav'
                original.append({'path': path, 'variant': 0, 'engine': 'fixture',
                    'text': 'Hey Nyx.' if label else ('Hey Mix.', 'Hey Next.', 'Hey Nick.')[i//2],
                    'label': label, 'sha256': str(i), 'split': 'training', 'offset': i*335, 'frames': 335})
                bank.append({'path': path, 'variant': 0, 'label': label, 'pair': i//2,
                    'source_sha256': str(i), 'end_frame': 335,
                    'feature_sha256': hashlib.sha256(x[i].tobytes()).hexdigest()})
            (root/'bank.json').write_text(json.dumps({'rows': bank}))
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            plan = {'bank_sha256': digest(root/'bank.json'), 'bank_data_sha256': digest(root/'bank.npz'),
                    'rows': 6, 'engines': ['fixture']}

            def index():
                (root/'features.json').write_text(json.dumps(original))
                plan['feature_index_sha256'] = digest(root/'features.json')

            index()
            features, labels, rows = load_bank(root, root, plan)
            np.testing.assert_array_equal(features, x)
            np.testing.assert_array_equal(labels, [0, 1, 0, 2, 0, 3])
            self.assertEqual(len(rows), 6)
            changed = x.copy(); changed[0, 0, 0] = 1
            changed.astype('<u2').tofile(root/'features.bin')
            with self.assertRaisesRegex(ValueError, 'exact cached crop'):
                load_bank(root, root, plan)
            x.astype('<u2').tofile(root/'features.bin')
            original.append({**original[0], 'path': 'reserved.wav', 'split': 'validation'})
            index()
            with self.assertRaisesRegex(ValueError, 'Training-only'):
                load_bank(root, root, plan)


if __name__ == '__main__':
    unittest.main()
