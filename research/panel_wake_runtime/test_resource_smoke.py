"""Reject persistent drift and hardware audio loss even with fast inference."""
from pathlib import Path
import tempfile
import unittest

from resource_smoke import diagnostics


class ResourceGateTest(unittest.TestCase):
    def measure(self, elapsed=8000, overflow=0, with_counters=True):
        lines = []
        for index in range(3):
            step = (index + 1) * 100
            lines.append(f'I ({10000 + index * elapsed}) panel.wake.embedding: step={step} '
                         'mean_us=50000 p50_us=49000 p95_us=60000 max_us=61000 over80ms=0')
            if with_counters:
                lines.append(f'I (0) panel.wake.embedding: capture step={step} '
                             f'rx_bytes={step * 1280 * 4} rx_overflows={17 + index * overflow}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'uart.log'
            path.write_text('\n'.join(lines))
            return diagnostics(path, capture_backlog_bytes=6784)

    def test_stable_capture_does_not_confuse_startup_loss_with_new_loss(self):
        result = self.measure()
        self.assertTrue(result['pipeline_cadence_passed'])
        self.assertTrue(result['capture_loss_passed'])

    def test_persistent_drift_fails_despite_each_window_fitting_three_percent(self):
        result = self.measure(elapsed=8180)
        self.assertTrue(result['inference_margin_passed'])
        self.assertTrue(result['pipeline_window_cadence_passed'])
        self.assertFalse(result['pipeline_cadence_passed'])

    def test_one_overflow_per_window_and_missing_observers_both_fail_loss_gate(self):
        self.assertFalse(self.measure(overflow=1)['capture_loss_passed'])
        self.assertFalse(self.measure(with_counters=False)['capture_loss_passed'])

    def test_partial_dma_tail_loss_without_queue_overflow_is_rejected(self):
        # IDF can replace a partially read buffer before its queue overflows.
        lines = []
        for i in range(5):
            step = (i + 1) * 100
            lines.append(f'capture step={step} rx_bytes={step * 1280 * 4 + i * 3520} rx_overflows=17')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'uart.log'
            path.write_text('\n'.join(lines))
            result = diagnostics(path, capture_backlog_bytes=6784)
        self.assertTrue(result['capture_queue_passed'])
        self.assertFalse(result['capture_balance_passed'])
        self.assertFalse(result['capture_loss_passed'])


if __name__ == '__main__':
    unittest.main()
