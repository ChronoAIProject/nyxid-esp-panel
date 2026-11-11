"""Check neutral logits and the expected first/second encoder-gradient behavior."""
import unittest

import numpy as np
import tensorflow as tf

from intent_phone_fit import targets, update_factory
from intent_phone_taps import phone_head
from phrase_ctc_readout import weights_sha


class InitializationTest(unittest.TestCase):
    def test_neutral_probabilities_are_input_and_seed_independent(self):
        x = np.random.default_rng(92).normal(size=(3, 67, 60)).astype(np.float32)
        a, b = phone_head('zero', 919), phone_head('zero', 111)
        self.assertEqual(a.count_params(), 610)
        self.assertEqual(weights_sha(a.weights), weights_sha(b.weights))
        np.testing.assert_array_equal(a(x), np.zeros((3, 67, 10), np.float32))
        np.testing.assert_allclose(tf.nn.softmax(a(x), axis=-1), .1, atol=1e-8, rtol=0)
        self.assertNotEqual(weights_sha(a.weights), weights_sha(phone_head('glorot', 919).weights))
        with self.assertRaises(ValueError):
            phone_head('invalid', 919)

    def test_first_step_preserves_encoder_then_gradients_connect(self):
        tf.keras.utils.set_random_seed(91)
        x = tf.keras.Input((67, 60))
        encoder = tf.keras.Model(x, tf.keras.layers.Dense(60, activation='tanh')(x))
        head = phone_head('zero', 919)
        before, head_before = weights_sha(encoder.weights), weights_sha(head.weights)
        _, update = update_factory(encoder, head,
            {'encoder_learning_rate': .0002, 'head_learning_rate': .003, 'global_clipnorm': 5})
        features = np.random.default_rng(94).normal(size=(4, 67, 60)).astype(np.float32)
        packed, lengths = targets(np.arange(4), 'phones')
        loss, _ = update(features, packed, lengths)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(before, weights_sha(encoder.weights))
        self.assertNotEqual(head_before, weights_sha(head.weights))
        update(features, packed, lengths)
        self.assertNotEqual(before, weights_sha(encoder.weights))


if __name__ == '__main__':
    unittest.main()
