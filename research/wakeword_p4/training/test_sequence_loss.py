"""Check the loss against independent float64 formulae and extreme logits."""
import unittest

import numpy as np
import tensorflow as tf

from sequence_loss import mean_probability_bce


class SequenceLossTest(unittest.TestCase):
    def value_gradient(self, labels, values):
        logits = tf.Variable(values, dtype=tf.float64)
        with tf.GradientTape() as tape:
            loss = mean_probability_bce(tf.constant(labels, dtype=tf.float64), logits)
        return loss.numpy(), tape.gradient(loss, logits).numpy()

    def test_independent_mixture_value_and_gradient(self):
        rng = np.random.default_rng(412)
        values = rng.uniform(-12, 12, (16, 5))
        labels = rng.uniform(0, 1, 16)
        probability = 1/(1+np.exp(-values))
        mean = probability.mean(axis=1)
        expected = -labels*np.log(mean)-(1-labels)*np.log1p(-mean)
        gradient = ((mean-labels)/(mean*(1-mean)))[:, None]*probability*(1-probability)/5
        loss, actual = self.value_gradient(labels, values)
        np.testing.assert_allclose(loss, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual, gradient, atol=1e-12, rtol=1e-12)

    def test_saturated_wrong_examples_retain_gradient(self):
        loss, gradient = self.value_gradient([0., 1.], [[1000.]*5, [-1000.]*5])
        np.testing.assert_allclose(loss, [1000., 1000.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient, [[.2]*5, [-.2]*5], atol=1e-12, rtol=0)

    def test_terminal_matches_builtin_logits_bce(self):
        values = np.array([[-1000., 0., 1000., 20., -20.]]).T
        labels = np.array([1., 0., 0., 1., 0.])
        loss, gradient = self.value_gradient(labels, values)
        z = tf.Variable(values, dtype=tf.float64)
        with tf.GradientTape() as tape:
            expected = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[:, None], logits=z)
        np.testing.assert_allclose(loss, expected.numpy()[:, 0], atol=1e-12, rtol=0)
        np.testing.assert_allclose(gradient, tape.gradient(expected, z), atol=1e-12, rtol=0)

    def test_compiled_float32_extremes_and_variable_length(self):
        @tf.function(input_signature=[tf.TensorSpec([None], tf.float32),
                                     tf.TensorSpec([None, None], tf.float32)])
        def compiled(labels, z):
            with tf.GradientTape() as tape:
                tape.watch(z)
                losses = mean_probability_bce(labels, z)
            return losses, tape.gradient(losses, z)
        for steps in (1, 5, 9):
            losses, gradients = compiled(tf.constant([0., 1.]), tf.constant([[80.]*steps, [-80.]*steps]))
            np.testing.assert_allclose(losses, [80., 80.], atol=1e-5, rtol=0)
            np.testing.assert_allclose(gradients, [[1/steps]*steps, [-1/steps]*steps], atol=1e-6, rtol=0)

    def test_not_mean_logits_or_mean_bce(self):
        values = np.array([[8., -3., -3., -3., -3.]])
        loss, _ = self.value_gradient([1.], values)
        mean_logits_bce = np.logaddexp(0, -values.mean(axis=1))
        mean_bce = np.logaddexp(0, -values).mean(axis=1)
        self.assertGreater(abs(loss[0]-mean_logits_bce[0]), .01)
        self.assertGreater(abs(loss[0]-mean_bce[0]), .01)


if __name__ == '__main__':
    unittest.main()
