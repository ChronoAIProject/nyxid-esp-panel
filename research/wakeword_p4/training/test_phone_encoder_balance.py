"""Check the encoder budget and its actual optimizer step independently."""
import unittest

import numpy as np

from phone_distillation import STEPS, VOCAB, balance_encoder_gradients, supervisor


class EncoderBalanceTest(unittest.TestCase):
    def fixture(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(81)
        x = tf.keras.Input((STEPS, 60))
        z = tf.keras.layers.Dense(60, use_bias=False, kernel_initializer='identity')(x)
        score = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier',
            kernel_initializer=tf.keras.initializers.Constant(.001))(tf.keras.layers.Flatten()(z))
        model = tf.keras.Model(x, score)
        model.compile(optimizer=tf.keras.optimizers.SGD(.001, global_clipnorm=5),
                      loss='binary_crossentropy')
        values = np.random.default_rng(44).normal(0, .01, (4, STEPS, 60)).astype(np.float32)
        y = np.array([[1., 1.], [0., 2.], [1., 3.], [0., 4.]], np.float32)
        weights = np.array([1., 5., 1., 5.], np.float32)
        q = np.zeros((4, STEPS, VOCAB), np.float32)
        q[..., 0] = 1
        q[:, 0, 0] = 0; q[:, 0, 1] = 1
        return model, values, y, weights, q

    def test_closed_form_vector_budget_preserves_heads_and_zero_cases(self):
        import tensorflow as tf
        b = [tf.constant([3., 4.]), tf.constant([2.]), tf.constant([0.])]
        p = [tf.constant([0., 100.]), tf.constant([0.]), tf.constant([30.])]
        mixed, scale, ratio = balance_encoder_gradients(b, p, {0}, .1, .25)
        np.testing.assert_allclose(mixed[0], [3., 5.25])
        np.testing.assert_array_equal(mixed[1], [2.])
        np.testing.assert_array_equal(mixed[2], [3.])
        self.assertAlmostEqual(float(scale), .125)
        self.assertAlmostEqual(float(ratio), .25)
        for binary, phone, gain in (([tf.zeros_like(v) for v in b], p, .1),
                                    (b, [tf.zeros_like(v) for v in p], .1), (b, p, 0.)):
            values, _, relative = balance_encoder_gradients(binary, phone, {0}, gain, .25)
            self.assertTrue(all(np.isfinite(v).all() for v in values))
            self.assertLessEqual(float(relative), .250001)

    def test_real_ctc_step_matches_independent_numpy_gradient_combination(self):
        import tensorflow as tf
        model, x, y, weights, q = self.fixture()
        train = supervisor(model, q, .1, 'ctc', .25)
        full = train.auxiliary_model
        variables = full.trainable_variables
        original = [v.numpy().copy() for v in variables]
        with tf.GradientTape(persistent=True) as tape:
            probability, phones = full(x, training=True)
            binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], probability)*weights)
            phone = train.phone_loss(tf.constant(q), phones)
        gradients = []
        for loss in (binary, phone):
            values = tape.gradient(loss, variables)
            gradients.append([np.asarray(g, dtype=np.float64) if g is not None
                              else np.zeros(tuple(v.shape), np.float64)
                              for g, v in zip(values, variables, strict=True)])
        del tape
        classifier = {id(v) for v in model.get_layer('classifier').trainable_variables}
        main = {id(v) for v in model.trainable_variables}
        encoder = [i for i, v in enumerate(variables) if id(v) in main-classifier]
        bg, pg = gradients
        bn = np.linalg.norm(np.concatenate([bg[i].ravel() for i in encoder]))
        pn = .1*np.linalg.norm(np.concatenate([pg[i].ravel() for i in encoder]))
        scale = min(1., .25*bn/pn)
        expected = [b+.1*p*(scale if i in encoder else 1.) for i, (b, p) in enumerate(zip(bg, pg))]
        clip = min(1., 5/np.linalg.norm(np.concatenate([v.ravel() for v in expected])))
        train(x, y, weights)
        for before, gradient, current in zip(original, expected, variables, strict=True):
            np.testing.assert_allclose(current.numpy(), before-.001*gradient*clip, rtol=1e-6, atol=2e-6)
        stats = train.balance_metadata()
        self.assertEqual(stats['steps'], 1)
        self.assertAlmostEqual(stats['minimum_scale'], scale, places=6)
        self.assertLessEqual(stats['maximum_scaled_encoder_ratio'], .250001)

    def test_zero_gain_retains_binary_optimizer_and_rejects_invalid_caps(self):
        model, x, y, weights, q = self.fixture()
        control, _, _, _, _ = self.fixture()
        control.set_weights(model.get_weights())
        train = supervisor(model, q, 0., 'ctc', .25)
        train(x, y, weights)
        control.train_on_batch(x, y[:, :1], sample_weight=weights)
        for a, b in zip(model.get_weights(), control.get_weights(), strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)
        self.assertFalse(train.balance_metadata()['enabled'])
        for cap in (0., 2., float('nan')):
            with self.assertRaises(ValueError):
                supervisor(model, q, .1, 'ctc', cap)


if __name__ == '__main__':
    unittest.main()
