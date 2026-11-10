"""Capture conservation should catch losses without a DMA overflow signal."""
import unittest

from measure_idle_headroom import capture_audit


def log(extra_rx_per_window=0, missing_window=False):
    lines = []
    # 167 detector blocks of 480 samples and 313 I2S blocks of 256 samples.
    # The pending wake input falls 32 samples per window. The first DMA
    # count is near uint32 wrap and a startup overflow is already recorded.
    for n in range(5):
        if missing_window and n == 2:
            continue
        rx = (0xfffff000 + n * (320512 + extra_rx_per_window)) % 2**32
        lines.append(
            f'panel.wake.timing: calls=167 pcm_samples=80160 wall_us=5008000 '
            f'detect_us=430000 max_us=9000 stack_free=11856 end_us={20_000_000 + n * 5008000} '
            f'read_bytes=320512 pending_samples={256 - n * 32} rx_bytes={rx} rx_overflows=826')
    return '\n'.join(lines).encode()


class IdleCaptureAuditTest(unittest.TestCase):
    def test_accepts_contiguous_capture_across_counter_wrap(self):
        self.assertTrue(capture_audit(log(), 6784, 20)['passed'])

    def test_rejects_cumulative_loss_even_with_no_overflow_or_cadence_drift(self):
        result = capture_audit(log(extra_rx_per_window=1920), 6784, 20)
        self.assertFalse(result['passed'])
        self.assertEqual(result['intervals'][-1]['unaccounted_rx_bytes_since_start'], 7680)

    def test_rejects_missing_uart_window(self):
        self.assertFalse(capture_audit(log(missing_window=True), 6784, 10)['passed'])
