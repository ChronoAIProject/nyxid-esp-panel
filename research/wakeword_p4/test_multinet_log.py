"""UART timeouts and offline recovery must preserve complete experiment records."""
import copy
import json
import unittest

from audit_multinet_log import audit, parse_log, reconstruct, sha
from run_multinet_probe import UartLines, record_line, validate_result


class MultiNetLogTest(unittest.TestCase):
    def setUp(self):
        self.manifest = {'command_arms': [{'name': 'target', 'commands': ['HEY NICKS']}],
            'sources': [{'clip': 0, 'samples': 513, 'label': 1},
                        {'clip': 1, 'samples': 512, 'label': 0}],
            'model_raw_bytes': 10, 'fixtures_raw_bytes': 20, 'cases': 2,
            'all_sources_training_only': True, 'audio_speed_changed': False,
            'plan_sha256': 'plan', 'detector_threshold': .9, 'detector_timeout_ms': 8000}
        self.value = {'ready': {'chunk': 512, 'rate': 16000, 'model_expanded_bytes': 10,
                               'fixtures_expanded_bytes': 20},
            'clips': [{'arm': 0, 'clip': clip, 'samples': samples, 'target_events': 0,
                       'other_events': 0, 'timeouts': 0, 'first_target_end_sample': -1}
                      for clip, samples in ((0, 513), (1, 512))],
            'timing': {'calls': 3, 'p50_us': 1, 'p95_us': 2, 'max_us': 3},
            'done': {'completed': True, 'cases': 2}}
        self.raw = b''.join((marker + ' ' + json.dumps(value) + '\r\n').encode()
            for marker, value in [('MN_READY', self.value['ready']),
                *[('MN_CLIP', r) for r in self.value['clips']],
                ('MN_TIMING', self.value['timing']), ('MN_DONE', self.value['done'])])

    def test_every_uart_boundary_and_empty_timeout_preserves_all_records(self):
        for boundary in range(len(self.raw) + 1):
            lines, result = UartLines(), {}
            for chunk in (self.raw[:boundary], b'', self.raw[boundary:]):
                for raw in lines.feed(chunk):
                    record_line(raw.decode().strip(), result)
            self.assertEqual(result, self.value)
            self.assertEqual(lines.pending, b'')

    def test_unterminated_done_is_not_accepted(self):
        lines = UartLines()
        result = {'clips': []}
        for raw in lines.feed(self.raw.rstrip(b'\r\n')):
            record_line(raw.decode().strip(), result)
        self.assertNotIn('done', result)
        with self.assertRaises(ValueError):
            validate_result(result, self.manifest)

    def test_duplicate_summary_or_clip_and_missing_calls_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_log(self.raw + b'MN_DONE {"completed":true,"cases":2}\n')
        for change in ('duplicate_clip', 'missing_call'):
            value = copy.deepcopy(self.value)
            if change == 'duplicate_clip':
                value['clips'].append(value['clips'][0])
            else:
                value['timing']['calls'] -= 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_result(value, self.manifest)

    def test_bounded_line_cannot_grow_forever(self):
        lines = UartLines(limit=4)
        self.assertEqual(lines.feed(b'123'), [])
        with self.assertRaises(ValueError):
            lines.feed(b'45')

    def artifacts(self):
        broken = self.raw.replace(b'MN_CLIP', b'M\nN_CLIP', 1)
        original = parse_log(broken)
        manifest = json.dumps(self.manifest).encode()
        original.update(vectors=self.manifest, vectors_manifest_sha256=sha(manifest),
            probe_bin_sha256='probe', restore_bin_sha256='restore',
            error={'type': 'ValueError', 'message': 'Missing, duplicate or unexpected MultiNet clip results'},
            restored={'original_service_verified': True, 'hi_esp_listening': True, 'nyx_state': 'up'})
        return broken, json.dumps(original).encode(), manifest

    def test_explicit_split_recovery_preserves_json_and_rechecks_coverage(self):
        broken, original, manifest = self.artifacts()
        repaired, repairs = reconstruct(broken)
        self.assertEqual(repaired, self.raw)
        self.assertEqual(len(repairs), 1)
        result = audit(broken, original, manifest, sha(broken), 1)
        self.assertEqual(result['clips'], self.value['clips'])
        self.assertEqual(result['reconstruction']['original_parsed_clips'], 1)
        self.assertTrue(result['reconstruction']['offline_coverage_validation_passed'])
        self.assertFalse(result['accepted'])

    def test_changed_identity_or_unexpected_repair_count_is_rejected(self):
        broken, original, manifest = self.artifacts()
        for expected, count in (('another-log', 1), (sha(broken), 0)):
            with self.subTest(expected=expected, count=count), self.assertRaises(ValueError):
                audit(broken, original, manifest, expected, count)

    def test_recovery_does_not_conceal_payload_damage(self):
        broken, original, manifest = self.artifacts()
        damaged = broken.replace(b'"samples": 513', b'"samples": 511')
        with self.assertRaises(ValueError):
            audit(damaged, original, manifest, sha(damaged), 1)
        for bad in (b'M\nN_CLIP {broken}\n', b'M\nN_CLIP {}'):
            with self.subTest(bad=bad), self.assertRaises((ValueError, KeyError)):
                reconstruct(bad)


if __name__ == '__main__':
    unittest.main()
