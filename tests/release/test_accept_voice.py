"""Fault controls for a harness that opens a real microphone/call."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    'accept_voice', Path(__file__).resolve().parents[2] / 'tools' / 'accept_voice.py')
voice = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voice)


class VoiceCleanupTests(unittest.TestCase):
    def harness(self, scenario='full'):
        run = voice.Acceptance(SimpleNamespace(scenario=scenario))
        run.initial = {'wake': {'enabled': True, 'detections': 1}, 'microphone_muted': False}
        run.session = 'our-test-session'
        run.status = Mock(return_value={'session_id': run.session, 'busy': True,
                                        'microphone_muted': False,
                                        'wake': {'initiated_call': True, 'detections': 2}})
        run.rows = Mock(return_value=[])
        run.proxy, run.wait = Mock(), Mock()
        run.report['passed'] = True
        return run

    def test_history_failure_does_not_skip_hangup(self):
        run = self.harness()
        run.rows.side_effect = OSError('history unavailable')
        run.cleanup()
        run.proxy.assert_called_once_with('/voice/stop', {})
        self.assertFalse(run.report['passed'])
        self.assertIn('history unavailable', str(run.report['cleanup_error']))

    def test_hangup_failure_does_not_skip_saved_wake_restoration(self):
        run = self.harness('boundaries')
        def action(path, body):
            if path == '/voice/stop':
                raise OSError('stop unavailable')
        run.proxy.side_effect = action
        run.cleanup()
        run.proxy.assert_any_call('/voice/wake', {'enabled': True})
        self.assertFalse(run.report['passed'])

    def test_cleanup_does_not_hang_up_another_session(self):
        run = self.harness('remote')
        run.remote_id = 'our-request'
        run.status.return_value.update(session_id='someone-else', remote_request_id='another-request')
        run.cleanup()
        run.proxy.assert_not_called()
        self.assertTrue(run.report['passed'])

    def test_previous_error_does_not_abort_before_wake_detection(self):
        run = self.harness()
        run.session = None
        run.status.side_effect = [
            {'state': 'error', 'active': False, 'wake': {'detections': 1}},
            {'state': 'active', 'active': True, 'wake': {'detections': 2}},
        ]
        with patch.object(voice.time, 'sleep'):
            self.assertTrue(run.wait(lambda v: v['active'])['active'])

    def test_current_attempt_error_is_reported(self):
        run = self.harness()
        run.session = None
        run.status.return_value = {'state': 'error', 'active': False,
                                   'message': 'provider rejected', 'wake': {'detections': 2}}
        with self.assertRaisesRegex(RuntimeError, 'provider rejected'):
            run.wait(lambda v: v['active'])


if __name__ == '__main__':
    unittest.main()
