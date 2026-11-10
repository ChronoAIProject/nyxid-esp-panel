"""Protect float-window phase and signed8bit quantization diagnostics."""
import unittest

import numpy as np

from diagnose_backbone_precision import float_window, quantize


class BackbonePrecisionTests(unittest.TestCase):
    def test_first_full_context_ends_at80_and_uses_last76(self):
        frames = np.repeat(np.arange(88)[:, None], 32, axis=1)
        blocks = [frames[i:i + 8] for i in range(0, 88, 8)]
        self.assertIsNone(float_window(blocks[:9]))
        np.testing.assert_array_equal(float_window(blocks[:10])[0, :, 0, 0], np.arange(4, 80))
        np.testing.assert_array_equal(float_window(blocks)[0, :, 0, 0], np.arange(12, 88))

    def test_saturation_is_counted_without_byte_wrap(self):
        values, count = quantize(np.asarray([-200., -128., 0., 127., 200.]), 1., 0)
        np.testing.assert_array_equal(values, [-128, -128, 0, 127, 127])
        self.assertEqual(count, 2)


if __name__ == '__main__':
    unittest.main()
