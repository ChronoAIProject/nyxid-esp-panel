import unittest

import numpy as np
from lexical_domain import engine_labels, reverse_gradient


class LexicalDomainTest(unittest.TestCase):
    def test_reversal_changes_feature_gradient_but_preserves_classifier_learning(self):
        import tensorflow as tf
        features = tf.Variable([[.2, -.4], [.7, .1]], dtype=tf.float32)
        kernel = tf.Variable([[.4, -.7], [.6, .3]], dtype=tf.float32)
        labels = tf.constant([0, 1])
        with tf.GradientTape() as ordinary:
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, features @ kernel, from_logits=True))
        baseline = ordinary.gradient(loss, [features, kernel])
        with tf.GradientTape() as adversarial:
            transformed = reverse_gradient(features, .05)
            new_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, transformed @ kernel, from_logits=True))
        changed = adversarial.gradient(new_loss, [features, kernel])
        np.testing.assert_array_equal(transformed.numpy(), features.numpy())
        np.testing.assert_array_equal(new_loss.numpy(), loss.numpy())
        np.testing.assert_allclose(changed[0].numpy(), -.05 * baseline[0].numpy(), atol=1e-8)
        np.testing.assert_allclose(changed[1].numpy(), baseline[1].numpy(), atol=1e-8)

    def test_engine_labels_bind_source_and_binary_role(self):
        rows = {'a': {'source_sha256': 'source-a', 'label': 1, 'engine': 'piper'},
                'b': {'source_sha256': 'source-b', 'label': 0, 'engine': 'qwen'}}
        labels = engine_labels([('a', 'source-a', 0), ('b', 'source-b', 2)], rows)
        np.testing.assert_array_equal(labels, [4, 5])
        self.assertEqual(labels.dtype, np.int32)
        for draw in [('a', 'wrong-source', 0), ('a', 'source-a', 1)]:
            with self.assertRaisesRegex(ValueError, 'source or binary role'):
                engine_labels([draw], rows)
        rows['a']['engine'] = 'unknown'
        with self.assertRaises(ValueError):
            engine_labels([('a', 'source-a', 0)], rows)

    def test_unplanned_reversal_scale_rejected(self):
        import tensorflow as tf
        with self.assertRaisesRegex(ValueError, 'fixed'):
            reverse_gradient(tf.constant([1.]), .3)


if __name__ == '__main__':
    unittest.main()
