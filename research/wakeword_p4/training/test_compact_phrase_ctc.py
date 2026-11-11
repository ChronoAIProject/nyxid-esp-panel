"""Check fixed compact targets, binary roles and the shared training loss."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compact_phrase_ctc import Sampler
from phone_distillation import supervisor
from phrase_ctc_readout import targets, weights_sha


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompactTrainingTest(unittest.TestCase):
    def test_original_labels_and_identical_paired_draws(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            x = np.stack([np.full((335, 40), i, np.uint16) for i in range(6)])
            x.astype('<u2').tofile(root/'features.bin')
            np.savez(root/'bank.npz', features=x)
            original, bank = [], []
            for i in range(6):
                label = 1-i % 2
                original.append({'path': f'{i}.wav', 'variant': 0, 'engine': 'fixture',
                    'text': 'Hey Nyx.' if label else ('Hey Mix.', 'Hey Next.', 'Hey Nick.')[i//2],
                    'label': label, 'sha256': str(i), 'split': 'training', 'offset': i*335, 'frames': 335})
                bank.append({'path': f'{i}.wav', 'variant': 0, 'label': label, 'pair': i//2,
                    'source_sha256': str(i), 'end_frame': 335,
                    'feature_sha256': hashlib.sha256(x[i].tobytes()).hexdigest()})
            (root/'features.json').write_text(json.dumps(original))
            (root/'bank.json').write_text(json.dumps({'rows': bank}))
            plan = {'source_sha256': {}, 'compact_bank': {
                'feature_index_sha256': digest(root/'features.json'), 'bank_sha256': digest(root/'bank.json'),
                'bank_data_sha256': digest(root/'bank.npz'), 'rows': 6, 'engines': ['fixture']}}
            (root/'plan.json').write_text(json.dumps(plan))
            left, right = [Sampler(root, root, root/'plan.json') for _ in range(2)]
            np.testing.assert_array_equal(left.targets[:, :2].argmax(-1),
                                          targets(np.array([0, 1, 0, 2, 0, 3])))
            self.assertTrue((left.targets[:, 2:].argmax(-1) == 0).all())
            for _ in range(3):
                a, b = [np.zeros((128, 335, 40), np.uint16) for _ in range(2)]
                ids = left.apply(a)[:, 0].astype(int)
                np.testing.assert_array_equal(ids, right.apply(b)[:, 0])
                np.testing.assert_array_equal(a, b)
                for slot in range(8):
                    positive, negative = ids[slot]-1, ids[slot+32]-1
                    self.assertEqual(positive % 2, 0)
                    self.assertEqual(negative, positive+1)
                    np.testing.assert_array_equal(a[slot], x[positive])
                    np.testing.assert_array_equal(a[slot+32], x[negative])
                self.assertTrue((ids[8:32] == 0).all())
                self.assertTrue((ids[40:] == 0).all())
            self.assertEqual(left.metadata(), right.metadata())
            self.assertEqual(left.metadata()['vocabulary'], 6)
            self.assertEqual(left.metadata()['draws'], 24)
            plan['source_sha256']['compact_phrase_ctc.py'] = 'changed'
            (root/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, 'source differs'):
                Sampler(root, root, root/'plan.json')

    def test_compact_ctc_matches_independent_loss_and_keeps_scalar_graph(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(17)
        x = tf.keras.Input((335, 40))
        z = tf.keras.layers.Conv1D(60, 3, strides=5, activation='tanh')(x)
        probability = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(
            tf.keras.layers.Flatten()(z))
        model = tf.keras.Model(x, probability)
        model.compile(optimizer=tf.keras.optimizers.Adam(.0002, global_clipnorm=5),
                      loss='binary_crossentropy')
        parameters = model.count_params()
        packed = np.zeros((4, 67), np.int32)
        packed[:, :2] = targets(np.arange(4))
        q = np.eye(6, dtype=np.float32)[packed]
        train = supervisor(model, q, .1, 'ctc')
        data = np.random.default_rng(11).normal(size=(4, 335, 40)).astype(np.float32)
        y = np.array([[1, 1], [0, 2], [0, 3], [0, 4]], np.float32)
        weights = np.array([1, 5, 5, 5], np.float32)
        p, logits = train.auxiliary_model(data)
        expected = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], p)*weights)
        ctc = tf.reduce_mean(tf.nn.ctc_loss(targets(np.arange(4)), logits, [2]*4, [67]*4,
                                          logits_time_major=False, blank_index=0))/2
        expected += .1*ctc
        before = weights_sha(model.weights)
        observed = float(train(data, y, weights))
        self.assertAlmostEqual(observed, float(expected), places=5)
        self.assertNotEqual(weights_sha(model.weights), before)
        self.assertEqual(model.count_params(), parameters)
        self.assertEqual(model.output_shape, (None, 1))
        self.assertEqual(train.auxiliary_model.output_shape[1], (None, 67, 6))
        with self.assertRaises(ValueError):
            model.get_layer('training_phone_logits')


if __name__ == '__main__':
    unittest.main()
