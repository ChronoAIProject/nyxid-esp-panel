"""Variable-length CTC math, fold target semantics and disjoint Adam updates."""
import itertools
import unittest

import numpy as np
import tensorflow as tf

from intent_phone_fit import targets, ctc_loss, hypothesis_scores, unique_bank, update_factory
from phrase_ctc_readout import weights_sha


class IntentTest(unittest.TestCase):
    def test_known_shared_phone_targets_and_padding(self):
        labels, lengths = targets(np.arange(4), 'phones')
        np.testing.assert_array_equal(lengths, [6, 6, 7, 5])
        np.testing.assert_array_equal(labels, [[1, 2, 3, 4, 5, 6, 0],
            [1, 2, 7, 4, 5, 6, 0], [1, 2, 3, 8, 5, 6, 9], [1, 2, 3, 4, 5, 0, 0]])
        words, word_lengths = targets(np.array([3, 1]), 'words')
        np.testing.assert_array_equal(words, [[1, 5], [1, 3]])
        np.testing.assert_array_equal(word_lengths, [2, 2])
        for invalid in (np.array([4]), np.array([-1]), np.array([1.]), np.zeros((2, 2), np.int32)):
            with self.assertRaises(ValueError):
                targets(invalid, 'phones')
        with self.assertRaises(ValueError):
            targets(np.array([1]), 'unknown')

    def test_variable_lengths_match_exhaustive_ctc_paths_with_repeat(self):
        logits = np.random.default_rng(20).normal(size=(2, 5, 3)).astype(np.float32)
        probabilities = np.exp(logits.astype(np.float64)-logits.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        references = [[1, 2, 1], [1, 1]]
        expected = []
        for p, reference in zip(probabilities, references, strict=True):
            probability = 0.
            for path in itertools.product(range(3), repeat=5):
                collapsed = [v for i, v in enumerate(path) if v and (i == 0 or v != path[i-1])]
                if collapsed == reference:
                    probability += np.prod([p[t, v] for t, v in enumerate(path)])
            expected.append(-np.log(probability))
        actual = ctc_loss([[1, 2, 1], [1, 1, 0]], [3, 2], logits)
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_hypothesis_batching_matches_individual_unequal_length_losses(self):
        logits = np.random.default_rng(12).normal(size=(3, 9, 10)).astype(np.float32)
        actual = hypothesis_scores(logits, 'phones')
        references, lengths = targets(np.arange(4), 'phones')
        expected = np.empty((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = -tf.nn.ctc_loss(references[j:j+1], logits[i:i+1],
                    lengths[j:j+1], [9], logits_time_major=False, blank_index=0)[0]
        # Separate TF float32 batch kernels differ by up to two ulps here.
        np.testing.assert_allclose(actual, expected, atol=4e-6, rtol=0)

    def test_unique_crops_remove_cartesian_replication_and_reject_conflicts(self):
        features = np.arange(2*3*4, dtype=np.uint16).reshape(2, 3, 4)[[0, 1, 0]]
        labels = np.array([0, 1, 0], np.int32)
        row = {'engine': 'piper', 'sha256': 'a'*64, 'variant': 0}
        rows = [row, {**row, 'sha256': 'b'*64}, row]
        reduced, selected, metadata = unique_bank(features, labels, rows)
        np.testing.assert_array_equal(reduced, features[:2])
        np.testing.assert_array_equal(selected, [0, 1])
        self.assertEqual(metadata[0]['bank_rows'], [0, 2])
        for bad_labels, bad_rows in ((np.array([0, 1, 2]), rows),
                                    (labels, [row, rows[1], {**row, 'engine': 'other'}])):
            with self.assertRaises(ValueError):
                unique_bank(features, bad_labels, bad_rows)

    def test_joint_gradient_and_adam_use_per_example_target_lengths(self):
        tf.keras.utils.set_random_seed(15)
        inputs = tf.keras.Input((9, 3))
        states = tf.keras.layers.Dense(4, activation='tanh')(inputs)
        encoder = tf.keras.Model(inputs, states)
        classifier = tf.keras.layers.Dense(1)(tf.keras.layers.Flatten()(states))
        disconnected = tf.keras.Model(inputs, classifier)
        classifier_before = weights_sha(disconnected.layers[-1].weights)
        h = tf.keras.Input((9, 4))
        head = tf.keras.Model(h, tf.keras.layers.Dense(10)(h))
        head.set_weights([v*30 for v in head.get_weights()])
        data = tf.constant(np.random.default_rng(30).normal(size=(4, 9, 3)), tf.float32)
        packed, lengths = targets(np.arange(4), 'phones')
        variables = encoder.trainable_variables+head.trainable_variables
        with tf.GradientTape() as tape:
            logits = head(encoder(data))
            losses = tf.nn.ctc_loss(packed, logits, lengths, [9]*4,
                logits_time_major=False, blank_index=0)
            expected_loss = tf.reduce_mean(losses/tf.cast(lengths, tf.float32))
        gradients = tape.gradient(expected_loss, variables)
        norm = np.sqrt(sum(np.sum(g.numpy().astype(np.float64)**2) for g in gradients))
        self.assertGreater(norm, 5)
        expected_weights = []
        for index, (v, g) in enumerate(zip(variables, gradients, strict=True)):
            gradient = g.numpy()*min(1., 5/norm)
            rate = .0002 if index < len(encoder.trainable_variables) else .003
            alpha = rate*np.sqrt(.001)/.1
            expected_weights.append(v.numpy()-alpha*(.1*gradient)/(np.sqrt(.001*gradient**2)+1e-7))
        full, update = update_factory(encoder, head, {'encoder_learning_rate': .0002,
            'head_learning_rate': .003, 'global_clipnorm': 5.})
        loss, actual_norm = update(data, packed, lengths)
        self.assertAlmostEqual(float(loss), float(expected_loss), places=5)
        self.assertAlmostEqual(float(actual_norm), norm, places=4)
        for actual, expected in zip(variables, expected_weights, strict=True):
            np.testing.assert_allclose(actual.numpy(), expected, atol=2e-7, rtol=2e-6)
        self.assertEqual(weights_sha(disconnected.layers[-1].weights), classifier_before)


if __name__ == '__main__':
    unittest.main()
