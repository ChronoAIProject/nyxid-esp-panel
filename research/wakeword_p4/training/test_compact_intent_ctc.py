"""Independent CTC math and unchanged warmup/scalar graph for mixed-source loss."""
import unittest

import numpy as np
import tensorflow as tf

from conditional_ctc import compact_loss, losses
from ctc_reference import loss_and_gradient
from phone_distillation import supervisor
from phrase_ctc_readout import targets, weights_sha


def fixture():
    tf.keras.utils.set_random_seed(17)
    x = tf.keras.Input((335, 40))
    states = tf.keras.layers.Conv1D(60, 3, strides=5, activation='tanh')(x)
    model = tf.keras.Model(x, tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(
        tf.keras.layers.Flatten()(states)))
    model.compile(optimizer=tf.keras.optimizers.Adam(.0002, global_clipnorm=5), loss='binary_crossentropy')
    packed = np.zeros((4, 67), np.int32)
    packed[:, :2] = targets(np.arange(4))
    q = np.eye(6, dtype=np.float32)[packed]
    return model, q


class CompactIntentTest(unittest.TestCase):
    def test_word_likelihood_and_gradient_match_independent_ctc(self):
        values = np.random.default_rng(817).normal(size=(2, 7, 6)).astype(np.float32)
        intents = [0, 2]
        packed = np.zeros((2, 7), np.int32)
        packed[:, :2] = targets(np.array(intents))
        q = np.eye(6, dtype=np.float32)[packed]
        expected_loss, expected_gradient = [], []
        for x, truth in zip(values, intents, strict=True):
            reference = [loss_and_gradient(x, row.tolist()) for row in targets(np.arange(4))]
            score = -np.array([v[0] for v in reference])
            p = np.exp(score-score.max()); p /= p.sum()
            expected_loss.append(-np.log(p[truth])); p[truth] -= 1
            expected_gradient.append(-np.einsum('c,ctv->tv', p, [v[1] for v in reference])/2)
        variable = tf.Variable(values)
        with tf.GradientTape() as tape:
            value = compact_loss(q, variable)
        self.assertAlmostEqual(float(value), float(np.mean(expected_loss)), places=6)
        np.testing.assert_allclose(tape.gradient(value, variable), expected_gradient, atol=1e-6, rtol=0)
        self.assertEqual(float(compact_loss(q[:0], variable[:0])), 0)
        self.assertEqual(tuple(losses(variable[:0], [], targets(np.arange(4)), [2]*4).shape), (0,))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            compact_loss(np.roll(q, 1, axis=-1), variable)

    def test_zero_gain_replays_legacy_update_and_keeps_scalar_graph(self):
        data = np.random.default_rng(18).normal(size=(4, 335, 40)).astype(np.float32)
        y = np.array([[1, 1], [0, 2], [0, 3], [0, 4]], np.float32)
        weights = np.array([1, 5, 5, 5], np.float32)
        results = []
        for gain in (None, 0):
            model, q = fixture()
            count = model.count_params()
            train = supervisor(model, q, .1, 'ctc', intent_gain=gain)
            loss = float(train(data, y, weights))
            results.append((loss, weights_sha(model.weights), weights_sha(model.optimizer.variables),
                            weights_sha(train.auxiliary_model.weights)))
            self.assertEqual(model.count_params(), count)
            self.assertEqual(model.output_shape, (None, 1))
            with self.assertRaises(ValueError):
                model.get_layer('training_phone_logits')
            if gain is not None:
                self.assertEqual(train.intent_metadata()['steps'], 1)
                self.assertEqual(train.intent_metadata()['paired_examples'], 4)
        self.assertEqual(results[0], results[1])

    def test_mixed_loss_adds_conditional_inside_existing_auxiliary_gain(self):
        model, q = fixture()
        train = supervisor(model, q, .1, 'ctc', intent_gain=1)
        data = np.random.default_rng(18).normal(size=(4, 335, 40)).astype(np.float32)
        y = np.array([[1, 1], [0, 2], [0, 3], [0, 4]], np.float32)
        weights = np.array([1, 5, 5, 5], np.float32)
        probability, logits = train.auxiliary_model(data)
        bce = float(tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], probability)*weights))
        ctc, conditional = [], []
        for row, truth in zip(np.asarray(logits), range(4), strict=True):
            score = -np.array([loss_and_gradient(row, ref.tolist())[0] for ref in targets(np.arange(4))])
            shifted = score-score.max()
            ctc.append(-score[truth]/2)
            conditional.append(np.log(np.exp(shifted).sum())-shifted[truth])
        expected = bce+.1*(np.mean(ctc)+np.mean(conditional))
        self.assertAlmostEqual(float(train(data, y, weights)), float(expected), places=5)
        for invalid in (-1, float('nan'), 2):
            with self.assertRaises(ValueError):
                supervisor(model, q, .1, 'ctc', intent_gain=invalid)

    def test_intent_gain_does_not_enter_head_warmup_or_binary_state(self):
        raw = np.random.default_rng(26).integers(0, 256, (4, 335, 40), dtype=np.uint16)
        reports = []
        for gain in (0, 1):
            model, q = fixture()
            before = weights_sha(model.weights)
            train = supervisor(model, q, .1, 'ctc', intent_gain=gain)
            reports.append(train.warmup(raw, 1, .003, batch_size=4))
            self.assertEqual(weights_sha(model.weights), before)
            self.assertEqual(train.intent_metadata()['steps'], 0)
        self.assertEqual(reports[0], reports[1])


if __name__ == '__main__':
    unittest.main()
