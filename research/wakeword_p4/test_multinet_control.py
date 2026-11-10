"""Control success must depend on actual command IDs and consumed PCM."""
import copy
import json
import unittest

from audit_multinet_control import audit
from audit_multinet_log import sha
import test_multinet_log as log_fixture


class MultiNetControlTest(unittest.TestCase):
    def setUp(self):
        base = log_fixture.MultiNetLogTest()
        base.setUp()
        self.vectors = copy.deepcopy(base.manifest)
        self.vectors.update(speaker_playback=False, synthesis_rate_override=False)
        self.vectors['command_arms'] = [{'name': 'stock', 'stock_grammar': True, 'commands': []}]
        for row, role, expected, crc in zip(self.vectors['sources'],
                ('known_command', 'wake_target'), ([14], []), (123, 456)):
            row.update(voice='training-voice', role=role, text='synthetic',
                       expected_command_ids=[expected], detect_pcm_crc32=crc)
        self.value = copy.deepcopy(base.value)
        self.value['arm_records'] = [{'arm': 0, 'stock_grammar': True}]
        self.value['clips'][0].update(other_events=1, input_crc32=123)
        self.value['clips'][1].update(input_crc32=456)
        self.event = {'arm': 0, 'clip': 0, 'end_sample': 512, 'state': 1, 'num': 1,
            'commands': [{'id': 14, 'phrase_id': 0, 'probability': .95}],
            'string': 'TURN ON THE LIGHT', 'raw_string': 'phonemes'}
        self.value['events'] = [self.event]
        self.value['decoder_records'] = [{**self.event, 'end_sample': 1024},
            {'arm': 0, 'clip': 1, 'end_sample': 512, 'state': 0, 'num': 0,
             'commands': [], 'string': '', 'raw_string': ''}]

    def artifacts(self):
        records = [('MN_READY', self.value['ready'])]
        for marker, key in (('MN_ARM', 'arm_records'), ('MN_EVENT', 'events'),
                            ('MN_DECODER', 'decoder_records'), ('MN_CLIP', 'clips')):
            records.extend((marker, row) for row in self.value[key])
        records += [('MN_TIMING', self.value['timing']), ('MN_DONE', self.value['done'])]
        uart = ''.join(marker + ' ' + json.dumps(row) + '\n' for marker, row in records).encode()
        manifest = json.dumps(self.vectors).encode()
        report = {**self.value, 'vectors': self.vectors, 'vectors_manifest_sha256': sha(manifest),
            'execution_completed': True, 'probe_bin_sha256': 'probe',
            'restored': {'original_service_verified': True, 'hi_esp_listening': True, 'nyx_state': 'up'}}
        return json.dumps(report).encode(), manifest, uart

    def test_known_command_ids_pass_without_accepting_wake_quality(self):
        value = audit(*self.artifacts())
        self.assertTrue(value['coverage_and_pcm_crc_verified'])
        self.assertTrue(value['arms'][0]['known_controls_passed'])
        self.assertFalse(value['accepted'])

    def test_wrong_command_is_failed_control_even_when_execution_completed(self):
        self.value['events'][0]['commands'][0]['id'] = 15
        value = audit(*self.artifacts())
        self.assertFalse(value['arms'][0]['known_controls_passed'])
        self.assertFalse(value['accepted'])

    def test_consumed_pcm_change_is_rejected(self):
        self.value['clips'][0]['input_crc32'] += 1
        with self.assertRaises(ValueError):
            audit(*self.artifacts())

    def test_event_accounting_or_position_cannot_be_hidden(self):
        for change in ('extra', 'missing', 'outside'):
            before = copy.deepcopy(self.value)
            if change == 'extra':
                self.value['events'].append(copy.deepcopy(self.event))
            elif change == 'missing':
                self.value['events'] = []
            else:
                self.value['events'][0]['end_sample'] = 1536
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit(*self.artifacts())
            self.value = before

    def test_missing_decoder_or_wrong_grammar_is_rejected(self):
        before = copy.deepcopy(self.value)
        self.value['decoder_records'].pop()
        with self.assertRaises(ValueError):
            audit(*self.artifacts())
        self.value = before
        self.value['arm_records'][0]['stock_grammar'] = False
        with self.assertRaises(ValueError):
            audit(*self.artifacts())


if __name__ == '__main__':
    unittest.main()
