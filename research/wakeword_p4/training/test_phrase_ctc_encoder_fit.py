"""Check gradient propagation, optimizer partitions and frozen classifier state."""
import unittest

import numpy as np
import tensorflow as tf

from phrase_ctc_encoder_fit import make_head, make_update, measure
from phrase_ctc_readout import targets, weights_sha


class EncoderFitTest(unittest.TestCase):
    def test_joint_step_matches_independent_gradient_and_adam_calculation(self):
        tf.keras.utils.set_random_seed(14)
        x = tf.keras.Input((5, 3))
        states = tf.keras.layers.Dense(4, activation='tanh')(x)
        encoder = tf.keras.Model(x, states)
        classifier = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(states))
        unused = tf.keras.Model(x, classifier)
        classifier_weights = unused.layers[-1].get_weights()
        head = make_head((5, 4), 919)
        head.set_weights([value*30 for value in head.get_weights()])
        encoder_copy = tf.keras.models.clone_model(encoder)
        encoder_copy.set_weights(encoder.get_weights())
        head_copy = make_head((5, 4), 919)
        head_copy.set_weights(head.get_weights())
        plan = {'encoder_learning_rate': .0002, 'learning_rate': .003}
        data = tf.constant(np.random.default_rng(2).normal(size=(4, 5, 3)), tf.float32)
        labels = tf.constant(targets(np.arange(4)))
        variables = encoder_copy.trainable_variables+head_copy.trainable_variables
        with tf.GradientTape() as tape:
            logits = head_copy(encoder_copy(data))
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels, logits, [2]*4, [5]*4,
                logits_time_major=False, blank_index=0))/2
        gradients = tape.gradient(loss, variables)
        # Independent first-step Adam formula, including a single shared clip.
        norm = np.sqrt(sum(np.sum(g.numpy().astype(np.float64)**2) for g in gradients))
        self.assertGreater(norm, 5., 'Exercise the shared clipping path')
        scale = min(1., 5./norm)
        expected = []
        for i, (variable, gradient) in enumerate(zip(variables, gradients, strict=True)):
            g = gradient.numpy()*scale
            rate = plan['encoder_learning_rate' if i < len(encoder_copy.weights) else 'learning_rate']
            alpha = rate*np.sqrt(1-.999)/(1-.9)
            expected.append(variable.numpy()-alpha*(.1*g)/(np.sqrt(.001*g*g)+1e-7))
        encoder_before = weights_sha(encoder.weights)
        head_before = weights_sha(head.weights)
        full, update = make_update(encoder, head, plan)
        measured_loss, measured_norm = update(data, labels)
        self.assertAlmostEqual(float(measured_loss), float(loss), places=6)
        self.assertAlmostEqual(float(measured_norm), norm, places=5)
        for actual, wanted in zip(encoder.weights+head.weights, expected, strict=True):
            np.testing.assert_allclose(actual.numpy(), wanted, atol=2e-7, rtol=2e-6)
        self.assertNotEqual(weights_sha(encoder.weights), encoder_before)
        self.assertNotEqual(weights_sha(head.weights), head_before)
        for actual, wanted in zip(unused.layers[-1].get_weights(), classifier_weights, strict=True):
            np.testing.assert_array_equal(actual, wanted)
        before = weights_sha(full.weights)
        result, scores, decoded = measure(full, np.asarray(data), np.arange(4), 2)
        self.assertEqual(weights_sha(full.weights), before)
        self.assertEqual(scores.shape, (4, 4))
        self.assertEqual(len(decoded), 4)
        self.assertEqual(result['rows'], 4)


if __name__ == '__main__':
    unittest.main()
