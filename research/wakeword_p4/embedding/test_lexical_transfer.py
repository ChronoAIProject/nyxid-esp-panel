"""Check probe isolation and weighting before inspecting real feature scores."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from probe_lexical_transfer import CLASSES, balanced_weights, fit_ridge, load_inputs, predict, sha


class LexicalTransferTest(unittest.TestCase):
    def test_class_engine_replication_does_not_change_fitted_model(self):
        rng = np.random.default_rng(4)
        labels = np.tile(np.arange(6), 4)
        engines = np.repeat(['a', 'b'], 12)
        x = rng.normal(size=(24, 9))
        original = fit_ridge(x, labels, engines, 1.)
        # Replicate every example of one class/engine group, preserving its distribution.
        ids = np.r_[np.arange(24), np.flatnonzero((labels == 0) & (engines == 'a'))]
        copied = fit_ridge(x[ids], labels[ids], engines[ids], 1.)
        for left, right in zip(original, copied):
            np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(balanced_weights(labels, engines).sum(), 1.)
        np.testing.assert_array_equal(predict(x, original), predict(x, copied))

    def test_incomplete_class_engine_group_fails(self):
        with self.assertRaisesRegex(ValueError, 'all six classes'):
            balanced_weights(np.array([0, 1]), np.array(['a', 'a']))

    def test_source_alias_dedup_and_heldout_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            feature_root = root / 'head-features-tts-v1'
            feature_root.mkdir()
            feature = feature_root / 'example.npy'
            np.save(feature, np.arange(32 * 96, dtype=np.int16).reshape(32, 96).astype(np.int8))
            rows, records = [], []
            for identity in ['b', 'a']:
                row = {'id': identity, 'source_sha256': 's' * 64, 'features_sha256': sha(feature),
                       'split': 'training', 'kind': 'tts', 'engine': 'example', 'label': 1, 'variant': 0,
                       'path': 'head-features-tts-v1/example.npy', 'frames': 32}
                rows.append(row)
                records.append({key: row[key] for key in ('id', 'source_sha256', 'features_sha256',
                                                         'engine', 'label', 'variant')} |
                               {'class': 0, 'minimum_end': 25, 'maximum_end': 28})
            index = feature_root / 'index.jsonl'
            index.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'classes': CLASSES, 'all_sources_training_only': True,
                                          'heldout_source_hash_disjoint': True, 'audio_speed_changed': False,
                                          'records': records, 'unique_source_recordings': 1}))
            plan = {'auxiliary_manifest_sha256': sha(manifest), 'feature_index_sha256': sha(index)}
            selected, arrays = load_inputs(root, manifest, plan)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]['id'], 'a')
            self.assertEqual(selected[0]['selected_end'], 26)
            np.testing.assert_array_equal(arrays[0], np.load(feature)[10:26] / 128)
            with index.open('a') as out:
                out.write(json.dumps(rows[0] | {'id': 'heldout', 'split': 'validation'}) + '\n')
            plan['feature_index_sha256'] = sha(index)
            with self.assertRaisesRegex(ValueError, 'training role'):
                load_inputs(root, manifest, plan)


if __name__ == '__main__':
    unittest.main()
