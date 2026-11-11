"""Independent blank-path math, cohort identity and opt-in training replay."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from background_ctc import empty_target_loss, mark_ambient, mask
from phone_distillation import supervisor
from phrase_ctc_readout import targets


def fixture(background_gain=None, positive_gain=None):
    tf.keras.utils.set_random_seed(17)
    x = tf.keras.Input((335, 40))
    z = tf.keras.layers.Conv1D(60, 3, strides=5, activation='tanh')(x)
    p = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(
        tf.keras.layers.Flatten()(z))
    model = tf.keras.Model(x, p)
    model.compile(optimizer=tf.keras.optimizers.Adam(.0002, global_clipnorm=5))
    packed = np.zeros((4, 67), np.int32)
    packed[:, :2] = targets(np.arange(4))
    return model, supervisor(model, np.eye(6, dtype=np.float32)[packed], .1,
                             'ctc', background_gain=background_gain, positive_gain=positive_gain)


class BackgroundTest(unittest.TestCase):
    def test_blank_path_probability_and_analytic_gradient(self):
        values = np.random.default_rng(18).normal(size=(2, 3, 4))
        logits = tf.Variable(values, dtype=tf.float64)
        shifted = values-values.max(-1, keepdims=True)
        probabilities = np.exp(shifted)/np.exp(shifted).sum(-1, keepdims=True)
        expected = -np.log(np.prod(probabilities[..., 0], axis=1)).mean()/3
        with tf.GradientTape() as tape:
            loss = empty_target_loss(logits)
        gradient = probabilities.copy()
        gradient[..., 0] -= 1
        gradient /= 6
        self.assertAlmostEqual(float(loss), expected, places=13)
        np.testing.assert_allclose(tape.gradient(loss, logits), gradient, atol=1e-15)

    def test_cohort_survives_shuffle_without_marking_tts_negatives(self):
        y = np.zeros((128, 2), np.float32)
        y[:32, 0] = 1
        y[:8, 1] = np.arange(1, 17, 2)
        y[32:40, 1] = np.arange(2, 17, 2)
        marked = mark_ambient(y, 48)
        order = np.random.default_rng(39).permutation(128)
        observed = np.asarray(mask(marked[order]))
        np.testing.assert_array_equal(observed, order >= 80)
        self.assertEqual(int(observed.sum()), 48)
        np.testing.assert_array_equal(marked[:, :2], y)
        self.assertTrue((marked[32:80, 2] == 0).all())

    def test_invalid_cohorts_are_rejected(self):
        for y, count in [(np.zeros((3, 1)), 1), (np.zeros((3, 2)), 0),
                         (np.zeros((3, 2)), True), (np.zeros((3, 2)), 4),
                         (np.array([[0, 0], [1, 0]]), 1),
                         (np.array([[0, 0], [0, 1]]), 1),
                         (np.array([[np.nan, 0], [0, 0]]), 1)]:
            with self.subTest(y=y, count=count), self.assertRaises(ValueError):
                mark_ambient(y, count)
        for y in [[[0, 0, .5]], [[0, 0, np.nan]], [[0, 0, np.inf]],
                  [[1, 0, 1]], [[0, 2, 1]], [[0, 0, 0]], [[0, 0]]]:
            with self.subTest(y=y), self.assertRaises(tf.errors.InvalidArgumentError):
                mask(tf.constant(y, tf.float32))
        for shape in [(0, 3, 4), (2, 0, 4), (2, 3, 1)]:
            with self.subTest(shape=shape), self.assertRaises(tf.errors.InvalidArgumentError):
                empty_target_loss(tf.zeros(shape))

    def test_combined_loss_matches_independent_computation(self):
        model, train = fixture(1.)
        x = np.random.default_rng(31).normal(size=(7, 335, 40)).astype(np.float32)
        y = mark_ambient(np.array([[1, 1], [0, 2], [0, 3], [0, 4],
                                  [0, 0], [0, 0], [0, 0]], np.float32), 2)
        weights = np.array([1, 5, 5, 5, 5, 5, 5], np.float32)
        p, logits = train.auxiliary_model(x)
        bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y[:, :1], p)*weights)
        ctc = tf.reduce_mean(tf.nn.ctc_loss(targets(np.arange(4)), logits[:4],
            [2]*4, [67]*4, logits_time_major=False, blank_index=0))/2
        background = -tf.reduce_mean(tf.math.log(tf.nn.softmax(logits[-2:], axis=-1)[..., 0]))
        self.assertAlmostEqual(float(train(x, y, weights)), float(bce+.1*(ctc+background)), places=5)
        self.assertEqual(train.background_metadata()['steps'], 1)
        self.assertEqual(train.background_metadata()['examples'], 2)
        self.assertEqual(model.output_shape, (None, 1))

    def test_explicit_zero_replays_default_updates_and_adam_state_exactly(self):
        old, old_train = fixture()
        new, new_train = fixture(0.)
        y = np.array([[1, 1], [0, 2], [0, 3], [0, 4], [0, 0], [0, 0]], np.float32)
        marked = mark_ambient(y, 2)
        weights = np.array([1, 5, 5, 5, 5, 5], np.float32)
        rng = np.random.default_rng(80)
        for step in range(4):
            x = rng.normal(size=(6, 335, 40)).astype(np.float32)
            self.assertEqual(float(old_train(x, y, weights)), float(new_train(x, marked, weights)))
            for a, b in zip(old_train.auxiliary_model.weights, new_train.auxiliary_model.weights, strict=True):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            for a, b in zip(old.optimizer.variables, new.optimizer.variables, strict=True):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertIsNone(old_train.background_metadata())
        self.assertEqual(new_train.background_metadata()['steps'], 4)
        self.assertEqual(new_train.background_metadata()['examples'], 8)

    def test_invalid_cli_fails_before_input_or_output_access(self):
        script = Path(__file__).with_name('train_stream.py')
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            common = [sys.executable, str(script), '--upstream', str(root/'missing'),
                      '--features', str(root/'missing'), '--output', str(root/'output')]
            compact = ['--compact-phrase-plan', str(root/'missing'), '--phone-gain', '.1']
            for extra in [[], compact+['--phone-background-gain', 'nan'],
                          compact+['--phone-background-gain', '-.1'],
                          compact+['--phone-background-gain', '1.1'],
                          compact+['--phone-gain', '0', '--phone-background-gain', '1']]:
                if not extra:
                    extra = ['--phone-background-gain', '1']
                run = subprocess.run(common+extra, capture_output=True, text=True)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertIn('Background CTC requires', run.stderr)
                self.assertNotIn('Traceback', run.stderr)
                self.assertFalse((root/'output').exists())


if __name__ == '__main__':
    unittest.main()
