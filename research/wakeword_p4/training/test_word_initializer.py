"""Verify imported groups, protected classifier and corrupted provenance."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from prepare_word_initializer import attach_head, restore_encoder
from phrase_ctc_readout import sha, weights_sha
from compare_phone_distillation import initializer_identities, warmup_identity


class WordInitializerTest(unittest.TestCase):
    def test_original_policy_keeps_full_warmup_identity(self):
        plan = {'init': {'sha256': 'a'*64}, 'arms': {'control': .1, 'phonetic': .1}}
        a = {'steps': 1000, 'cache': 'original', 'schedule_sha256': 'fixed'}
        b = {**a, 'cache': 'different'}
        self.assertEqual(initializer_identities(plan), {'control': 'a'*64, 'phonetic': 'a'*64})
        self.assertNotEqual(warmup_identity(a, plan), warmup_identity(b, plan))

    def test_explicit_initializer_study_compares_recipe_and_draws(self):
        plan = {'init': {'sha256': 'a'*64}, 'arms': {'control': .1, 'phonetic': .1},
            'arm_initializer_sha256': {'control': 'a'*64, 'phonetic': 'b'*64},
            'warmup': {'steps': 1000, 'seed': 1223}}
        a = {'steps': 1000, 'seed': 1223, 'cache': 'original', 'schedule_sha256': 'fixed'}
        b = {**a, 'cache': 'different'}
        self.assertEqual(warmup_identity(a, plan), warmup_identity(b, plan))
        self.assertNotEqual(warmup_identity(a, plan), warmup_identity({**b, 'schedule_sha256': 'changed'}, plan))
        self.assertNotEqual(warmup_identity(a, plan), warmup_identity({**b, 'seed': 4}, plan))
        for identities in (None, ['control', 'phonetic'],
                           {'control': 'a'*64}, {'control': 'b'*64, 'phonetic': 'b'*64},
                           {'control': 'a'*64, 'phonetic': 'bad'}):
            with self.assertRaises(ValueError):
                initializer_identities({**plan, 'arm_initializer_sha256': identities})

    def test_imports_encoder_without_overwriting_scalar_classifier(self):
        tf.keras.utils.set_random_seed(38)
        x = tf.keras.Input((335, 40))
        states = tf.keras.layers.Conv1D(60, 3, strides=5)(x)
        binary = tf.keras.Model(x, tf.keras.layers.Dense(1, name='classifier')(
            tf.keras.layers.Flatten()(states)))
        original = binary.get_weights()
        count = binary.count_params()
        encoder, head, full = attach_head(binary)
        before = weights_sha(encoder.weights)
        protected = weights_sha(binary.get_layer('classifier').weights)
        encoder.set_weights([np.full_like(v, .125) for v in encoder.get_weights()])
        expected_encoder = [v.copy() for v in encoder.get_weights()]
        expected = {'encoder_weights_before_sha256': before,
            'encoder_weights_after_sha256': weights_sha(encoder.weights),
            'head_weights_sha256': weights_sha(head.weights),
            'classifier_weights_sha256': protected, 'classifier_unchanged': True}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/'words.weights.h5'
            full.save_weights(checkpoint)
            expected['checkpoint_sha256'] = sha(checkpoint)
            binary.set_weights(original)
            restored, _, _ = restore_encoder(binary, checkpoint, expected)
            for actual, wanted in zip(restored.get_weights(), expected_encoder, strict=True):
                np.testing.assert_array_equal(actual, wanted)
            self.assertEqual(weights_sha(binary.get_layer('classifier').weights), protected)
            self.assertEqual(binary.count_params(), count)
            self.assertEqual(binary.output_shape, (None, 1))
            for key in ('checkpoint_sha256', 'encoder_weights_before_sha256',
                        'classifier_weights_sha256', 'encoder_weights_after_sha256',
                        'head_weights_sha256'):
                binary.set_weights(original)
                with self.assertRaises(ValueError):
                    restore_encoder(binary, checkpoint, {**expected, key: 'corrupt'})


if __name__ == '__main__':
    unittest.main()
