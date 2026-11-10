"""Keep acoustic misses, early events and target changes distinct."""
import unittest

from acoustic_wake_probe import event_window, verify_target


class AcousticWakeProbeTests(unittest.TestCase):
    def test_no_event_does_not_manufacture_a_latency(self):
        voice = {'wake': {'detections': 7}}
        self.assertIsNone(event_window(voice, voice, 2.4, 2.6, 2, 3))

    def test_premature_and_onset_uncertain_events_stay_explicit(self):
        a, b = {'wake': {'detections': 7}}, {'wake': {'detections': 8}}
        early = event_window(a, b, 1.5, 1.8, 2, 3)
        self.assertTrue(early['before_source_onset'])
        self.assertFalse(early['onset_straddled'])
        uncertain = event_window(a, b, 1.9, 2.1, 2, 3)
        self.assertTrue(uncertain['onset_straddled'])
        hit = event_window(a, b, 3.1, 3.4, 2, 3)
        self.assertAlmostEqual(hit['delay_from_estimated_keyword_end_ms'][0], 100)
        self.assertAlmostEqual(hit['delay_from_estimated_keyword_end_ms'][1], 400)

    def test_counter_reset_and_multiple_events_are_not_a_single_hit(self):
        for count in (6, 9):
            with self.assertRaises(ValueError):
                event_window({'wake': {'detections': 7}}, {'wake': {'detections': count}}, 2, 3, 2, 3)

    def test_correct_app_at_another_service_is_rejected(self):
        status = {'app_sha256': 'app', 'nyx': {'slug': 'dev', 'state': 'up'},
                  'ota': {'busy': False}, 'voice': {'wake': {'word': 'Hi ESP'}}}
        verify_target(status, 'app', 'Hi ESP', 'dev')
        with self.assertRaises(ValueError):
            verify_target(status, 'app', 'Hi ESP', 'different-device')
        with self.assertRaises(ValueError):
            verify_target(status, 'different-app', 'Hi ESP', 'dev')


if __name__ == '__main__':
    unittest.main()
