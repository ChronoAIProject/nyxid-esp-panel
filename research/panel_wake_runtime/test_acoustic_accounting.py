"""Ensure retained playback/poll records cannot conceal failed controls."""
import copy
import unittest

from audit_acoustic_accounting import audit_pair


class AcousticAccountingTest(unittest.TestCase):
    def setUp(self):
        source = dict(id='a', voice='v', kind='positive', category='positive',
            text='Hi ESP', path='a.wav', sha256='source', samples=64000,
            keyword_end=48000, source_onset_samples=32000)
        self.plan = {'arms': {'hi_esp': {'app_elf_sha256': 'app', 'trials': [source]}}}
        self.before = dict(arm='hi_esp', complete=True, plan_sha256='plan', app_elf_sha256='app',
            trials=[{**source, 'event': {'observed_interval_s': [3.0, 3.3]}, 'rearmed': True,
                'playback_conditions_unchanged': True, 'source_playback_completed': False,
                'playback_returncode_before_cleanup': None}])
        self.after = copy.deepcopy(self.before)
        self.after['trials'][0].update(event=None, source_playback_completed=True,
                                       playback_returncode_before_cleanup=0)
        self.uart = 'rst:0x1 (POWERON)\nMC Quantized wakenet9: fixed-model\n'

    def run_pair(self, before_uart=None, after_uart=None):
        return audit_pair(self.before, self.after,
            self.uart + 'I (42) panel.wake: Hi ESP detected; recording\n' if before_uart is None else before_uart,
            self.uart if after_uart is None else after_uart, self.plan, 'plan')

    def test_changed_control_is_retained_without_claiming_input_equivalence(self):
        value = self.run_pair()
        self.assertEqual(value['changed_trials'], [{'id': 'a', 'category': 'positive',
            'before_triggered': True, 'after_triggered': False}])
        self.assertEqual(value['arms']['before']['detected_trials_stopped_before_source_completion'], 1)
        self.assertFalse(value['captured_microphone_audio_compared'])
        self.assertFalse(value['accepted'])

    def test_failed_source_playback_cannot_count_as_detector_miss(self):
        self.after['trials'][0]['source_playback_completed'] = False
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_unreported_uart_event_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_pair(after_uart=self.uart + 'I (52) panel.wake: Hi ESP detected; recording\n')

    def test_changed_input_or_incomplete_arm_is_rejected(self):
        original = copy.deepcopy(self.after)
        for field, value in (('sha256', 'changed'), ('samples', 32000), ('id', 'b')):
            self.after = copy.deepcopy(original)
            self.after['trials'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_pair()
        self.after = copy.deepcopy(original)
        self.after['complete'] = False
        with self.assertRaises(ValueError):
            self.run_pair()

    def test_extra_boot_or_model_change_is_rejected(self):
        for extra in ('rst:0x3 (SW_RESET)\n', 'MC Quantized wakenet9: different-model\n'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.run_pair(after_uart=self.uart + extra)

    def test_boot_garbage_is_counted_but_runtime_corruption_is_rejected(self):
        value = self.run_pair(after_uart='\ufffd\ufffdboot\n' + self.uart)
        self.assertEqual(value['arms']['after']['uart_utf8_replacement_characters'], 2)
        with self.assertRaises(ValueError):
            self.run_pair(after_uart=self.uart + '\ufffd')


if __name__ == '__main__':
    unittest.main()
