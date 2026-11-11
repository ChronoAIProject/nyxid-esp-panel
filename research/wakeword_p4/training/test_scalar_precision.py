"""Verify actual float TFLite state, chunk boundaries and reset semantics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scalar_precision import Detector


class PrecisionAdapterTest(unittest.TestCase):
    def test_stateful_float_stream_matches_independent_recurrence(self):
        import tensorflow as tf

        class Toy(tf.Module):
            def __init__(self):
                self.state = tf.Variable(0., trainable=False)

            @tf.function(input_signature=[tf.TensorSpec((1, 3, 40), tf.float32)])
            def step(self, value):
                result = self.state*.5+tf.reduce_mean(value)*.01
                self.state.assign(result)
                return tf.reshape(result, (1, 1))

        toy = Toy()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([toy.step.get_concrete_function()], toy)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'toy.tflite'; path.write_bytes(converter.convert())
            detector = Detector(path)
            raw = np.random.default_rng(17).integers(0, 600, (772, 40), dtype=np.uint16)
            raw[:3] = 1000  # Exercises the original upper clipping boundary.
            state, expected = 0., []
            for at in range(0, len(raw)-2, 3):
                state = state*.5+np.clip(raw[at:at+3].astype(np.float32)/25.6, 0, 26).mean()*.01
                expected.append(state)
            observed = detector.predict(raw)
            np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=0)
            detector.reset()
            chunks = np.concatenate([detector.predict(raw[:768]), detector.predict(raw[768:])])
            np.testing.assert_array_equal(chunks, observed)
            detector.reset()
            np.testing.assert_array_equal(detector.predict(raw), observed)
            self.assertEqual(len(observed), 257)
            self.assertEqual(detector.predict(raw[:2]).shape, (0,))

    def test_wrong_streaming_stride_is_rejected(self):
        import tensorflow as tf

        @tf.function(input_signature=[tf.TensorSpec((1, 4, 40), tf.float32)])
        def wrong(value):
            return tf.reshape(tf.reduce_mean(value), (1, 1))

        converter = tf.lite.TFLiteConverter.from_concrete_functions([wrong.get_concrete_function()])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)/'wrong.tflite'; path.write_bytes(converter.convert())
            with self.assertRaisesRegex(ValueError, 'stride-three'):
                Detector(path)


if __name__ == '__main__':
    unittest.main()
