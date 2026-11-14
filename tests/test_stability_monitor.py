import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('panel_stability', Path(__file__).resolve().parents[1] / 'tools/monitor_stability.py')
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def status():
    return dict(nyx=dict(slug='test-board', agent_name='private-agent'), fw='0.6.27',
        app_sha256='a' * 64, uptime_s=1000,
        heap=dict(free=20000000, spiram_free=19000000, dma_free=200000,
                  dma_largest_free=100000, internal_free=300000, internal_largest_free=100000),
        voice=dict(active=False, busy=False, state='idle', wake=dict(enabled=True, listening=True),
                   session_id='private-session', message='private-transcript'),
        media=dict(mode='idle'), lcd=dict(mode='panel'), ota=dict(stage='idle'),
        net=dict(connected=True, wifi=dict(ssid='private-ssid', password='private-password')))


class StabilityMonitorTests(unittest.TestCase):
    def test_allowlist_excludes_private_response_fields(self):
        row = monitor.sample(status(), 'test-board', 1000000)
        data = json.dumps(row)
        for text in ('private-agent', 'private-session', 'private-transcript',
                     'private-ssid', 'private-password', 'test-board'):
            self.assertNotIn(text, data)
        self.assertEqual(set(row['heap']), set(monitor.HEAP))

    def test_wrong_device_and_invalid_heap_are_rejected(self):
        with self.assertRaises(monitor.IdentityMismatch):
            monitor.sample(status(), 'another-board', 1000000)
        invalid = status()
        invalid['heap']['internal_free'] = -1
        with self.assertRaises(ValueError):
            monitor.sample(invalid, 'test-board', 1000000)

    def rows(self, decline=False, separate_phases=False):
        rows = []
        for i in range(60):
            s = status()
            if decline:
                s['heap']['internal_free'] -= i * 256
            if separate_phases:
                s['lcd']['mode'] = 'panel' if i % 2 else 'camera'
                s['heap']['internal_free'] = 200000 if i % 2 else 100000
            s['uptime_s'] += i * 60
            rows.append(monitor.sample(s, 'test-board', 1000000 + i * 60))
        return rows

    def test_sustained_loss_is_a_suspicion_and_stable_memory_is_clear(self):
        self.assertFalse(any(t['suspected_retained_loss'] for t in monitor.analyze(self.rows())))
        trends = monitor.analyze(self.rows(decline=True))
        self.assertTrue(any(t['metric'] == 'internal_free' and t['suspected_retained_loss'] for t in trends))
        self.assertEqual(monitor.analyze(self.rows(decline=True)[:10]), [])

    def test_different_workloads_are_not_compared_as_a_leak(self):
        self.assertFalse(any(t['suspected_retained_loss'] for t in monitor.analyze(self.rows(separate_phases=True))))

    def test_image_allocations_are_separate_and_activity_counters_are_retained(self):
        rows = self.rows()
        for i, row in enumerate(rows):
            s = status()
            s['lcd']['temporary_image'] = dict(active=i >= 30, jpeg_bytes=100000 if i >= 30 else 0,
                                              width=800 if i >= 30 else 0, height=600 if i >= 30 else 0)
            s['voice']['wake'].update(detections=7, buffered_ms=320, transmitted_ms=640)
            s['media'].update(captures=8, failed_captures=1)
            s['heap']['internal_free'] -= 20000 if i >= 30 else 0
            rows[i] = monitor.sample(s, 'test-board', row['t'])
        self.assertFalse(any(t['suspected_retained_loss'] for t in monitor.analyze(rows)))
        self.assertEqual(rows[-1]['counters']['wake_detections'], 7)
        self.assertEqual(rows[-1]['counters']['camera_failures'], 1)

    def test_recent_history_survives_monitor_restart(self):
        with tempfile.TemporaryDirectory() as root:
            config = dict(status_url='http://unused/status', expected_slug='test-board', state_dir=root, interval_s=30)
            now = monitor.time.time()
            m = monitor.Monitor(config)
            for i in range(3):
                s = status()
                s['uptime_s'] += i * 30
                with patch.object(monitor.time, 'time', return_value=now - 60 + i * 30), patch.object(
                        monitor.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(s).encode())):
                    m.poll()
            restored = monitor.Monitor(config)
            self.assertEqual(list(restored.rows), list(m.rows))
            self.assertEqual(restored.last, m.last)

    def test_reboot_and_poll_failure_break_the_trend_window(self):
        with tempfile.TemporaryDirectory() as root:
            config = dict(status_url='http://unused/status', expected_slug='test-board', state_dir=root, interval_s=30)
            m = monitor.Monitor(config)
            s = status()
            with patch.object(monitor.time, 'time', return_value=1000000), patch.object(
                    monitor.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(s).encode())):
                self.assertEqual(m.poll()['health'], 'online')
            s['uptime_s'] = 2
            with patch.object(monitor.time, 'time', return_value=1000030), patch.object(
                    monitor.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(s).encode())):
                state = m.poll()
                self.assertEqual(state['observed_restarts'], 1)
                self.assertEqual(state['recent_samples'], 1)
            with patch.object(monitor.urllib.request, 'urlopen', side_effect=OSError('private-url')):
                state = m.poll()
                self.assertEqual(state['recent_samples'], 0)
                self.assertEqual(state['health'], 'unreachable_or_invalid')
            self.assertNotIn('private-url', (Path(root) / 'status.json').read_text())
            changed = monitor.Monitor(dict(config, expected_slug='new-board'))
            self.assertIsNone(changed.last)
            self.assertFalse(changed.rows)

    def test_sleep_gap_is_recorded_and_wrong_device_never_adds_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            m = monitor.Monitor(dict(status_url='http://unused/status', expected_slug='test-board', state_dir=root, interval_s=30))
            for timestamp, uptime in ((1000000, 1000), (1000600, 1600)):
                s = status()
                s['uptime_s'] = uptime
                with patch.object(monitor.time, 'time', return_value=timestamp), patch.object(
                        monitor.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(s).encode())):
                    m.poll()
            self.assertEqual(len(m.rows), 1)
            self.assertIn('monitor_gap', ''.join(p.read_text() for p in Path(root).glob('metrics-*.jsonl')))
            s['nyx']['slug'] = 'private-other-board'
            with patch.object(monitor.urllib.request, 'urlopen', return_value=io.BytesIO(json.dumps(s).encode())):
                self.assertEqual(m.poll()['health'], 'identity_mismatch')
            self.assertNotIn('private-other-board', ''.join(p.read_text() for p in Path(root).glob('metrics-*.jsonl')))

    def test_rotation_only_removes_own_expired_logs(self):
        with tempfile.TemporaryDirectory() as root:
            m = monitor.Monitor(dict(status_url='http://unused/status', expected_slug='test-board', state_dir=root, interval_s=30))
            day = datetime.now(timezone.utc) - timedelta(days=40)
            expired = Path(root) / ('metrics-' + day.strftime('%Y-%m-%d') + '.jsonl')
            expired.write_text('{}\n')
            unrelated = Path(root) / 'notes.txt'
            unrelated.write_text('keep')
            m.log(dict(kind='test', t=datetime.now(timezone.utc).timestamp()))
            self.assertFalse(expired.exists())
            self.assertEqual(unrelated.read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
