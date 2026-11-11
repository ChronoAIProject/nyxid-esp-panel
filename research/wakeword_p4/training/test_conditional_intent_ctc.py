"""Independent likelihood and gradient checks for the four-hypothesis objective."""
import unittest

import numpy as np
import tensorflow as tf

from ctc_reference import loss_and_gradient
from intent_phone_fit import conditional_intent_losses, targets, update_factory


class ConditionalTest(unittest.TestCase):
    def test_conditional_value_and_gradient_match_independent_reference(self):
        values = np.random.default_rng(317).normal(size=(2, 9, 10)).astype(np.float32)
        truth = np.array([1, 3], np.int32)
        references, lengths = targets(np.arange(4), 'phones')
        expected_loss, expected_gradient = [], []
        for x, label in zip(values, truth, strict=True):
            reference = [loss_and_gradient(x, row[:length].tolist())
                         for row, length in zip(references, lengths, strict=True)]
            score = -np.array([value[0] for value in reference])
            p = np.exp(score-score.max()); p /= p.sum()
            expected_loss.append(-np.log(p[label]))
            p[label] -= 1
            expected_gradient.append(-np.einsum('c,ctv->tv', p, [value[1] for value in reference]))
        variable = tf.Variable(values)
        with tf.GradientTape() as tape:
            losses = conditional_intent_losses(variable, truth)
            objective = tf.reduce_sum(losses)
        np.testing.assert_allclose(losses, expected_loss, atol=3e-6, rtol=0)
        np.testing.assert_allclose(tape.gradient(objective, variable), expected_gradient, atol=2e-6, rtol=0)

    def test_combined_objective_requires_intent_and_uses_declared_gain(self):
        tf.keras.utils.set_random_seed(23)
        x = tf.keras.Input((9, 3))
        encoder = tf.keras.Model(x, tf.keras.layers.Dense(4, activation='tanh')(x))
        h = tf.keras.Input((9, 4))
        head = tf.keras.Model(h, tf.keras.layers.Dense(10)(h))
        data = np.random.default_rng(71).normal(size=(4, 9, 3)).astype(np.float32)
        packed, lengths = targets(np.arange(4), 'phones')
        logits = head(encoder(data))
        generative = tf.reduce_mean(tf.nn.ctc_loss(packed, logits, lengths, [9]*4,
            logits_time_major=False, blank_index=0)/tf.cast(lengths, tf.float32))
        expected = generative+tf.reduce_mean(conditional_intent_losses(logits, np.arange(4)))
        plan = {'encoder_learning_rate': .0002, 'head_learning_rate': .003,
                'global_clipnorm': 5, 'conditional_ctc_gain': 1}
        _, update = update_factory(encoder, head, plan)
        with self.assertRaises(ValueError):
            update(data, packed, lengths)
        loss, norm = update(data, packed, lengths, np.arange(4, dtype=np.int32))
        self.assertAlmostEqual(float(loss), float(expected), places=5)
        self.assertTrue(np.isfinite(float(norm)))
        for invalid in (-1, float('nan')):
            with self.assertRaises(ValueError):
                update_factory(encoder, head, {**plan, 'conditional_ctc_gain': invalid})


if __name__ == '__main__':
    unittest.main()
