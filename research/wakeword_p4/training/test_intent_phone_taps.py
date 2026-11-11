"""Check true receptive fields and disjoint phone-tap updates."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import tensorflow as tf

from intent_phone_fit import targets, update_factory
from intent_phone_taps import encoder_tap
from phrase_ctc_readout import weights_sha


class TapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream = os.environ.get('WAKE_TAP_UPSTREAM')
        if not upstream or not (Path(upstream)/'microwakeword').is_dir():
            raise RuntimeError('Set WAKE_TAP_UPSTREAM to the pinned microWakeWord source')
        sys.path.insert(0, upstream)

    def test_actual_graph_right_endpoints_and_local_receptive_fields(self):
        from transfer import build_model
        binary, _ = build_model(335)
        # Positive fixed weights avoid a randomly dead ReLU obscuring support.
        binary.set_weights([np.full(v.shape, .01, np.float32) for v in binary.weights])
        for kind, extent in (('deep', 137), ('local', 41)):
            encoder = encoder_tap(binary, kind)
            self.assertEqual(encoder.output_shape, (None, 67, 60))
            for index in (0, 33, 66):
                x = tf.Variable(np.ones((1, 335, 40), np.float32))
                with tf.GradientTape() as tape:
                    objective = tf.reduce_sum(encoder(x)[:, index])
                gradient = tape.gradient(objective, x).numpy()[0]
                support = np.flatnonzero(np.any(gradient != 0, axis=1))
                right = 136+3*index
                np.testing.assert_array_equal(support, np.arange(right-extent+1, right+1))
        with self.assertRaises(ValueError):
            encoder_tap(binary, 'unknown')

    def test_local_update_changes_only_its_encoder_and_head(self):
        from transfer import build_model
        tf.keras.utils.set_random_seed(51)
        binary, _ = build_model(335)
        encoder = encoder_tap(binary, 'local')
        ids = {id(v) for v in encoder.trainable_variables}
        suffix = [v for v in binary.weights if id(v) not in ids]
        suffix_before, before = weights_sha(suffix), weights_sha(encoder.weights)
        h = tf.keras.Input((67, 60))
        head = tf.keras.Model(h, tf.keras.layers.Dense(10)(h))
        head_before = weights_sha(head.weights)
        _, update = update_factory(encoder, head,
            {'encoder_learning_rate': .0002, 'head_learning_rate': .003, 'global_clipnorm': 5})
        packed, lengths = targets(np.arange(4), 'phones')
        loss, _ = update(np.random.default_rng(52).uniform(0, 10, (4, 335, 40)).astype(np.float32),
                         packed, lengths)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(weights_sha(suffix), suffix_before)
        self.assertNotEqual(weights_sha(encoder.weights), before)
        self.assertNotEqual(weights_sha(head.weights), head_before)


if __name__ == '__main__':
    unittest.main()
