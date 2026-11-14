"""Constructed PCM/raw/UART tests. No learned model or acoustic input."""
import copy
import io
import json
from pathlib import Path
import struct
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
import wave

from audit_complete_selftest import audit
import prepare_complete_selftest as complete


def wav(pcm, *, rate=16000, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, 'wb') as source:
        source.setframerate(rate)
        source.setnchannels(channels)
        source.setsampwidth(width)
        source.writeframes(pcm)
    return output.getvalue()


class CompleteFixture(unittest.TestCase):
    def setUp(self):
        self.model = b'\0\0\0\0TFL3'+bytes(1016)  # Header-shaped, not inferable.
        self.pcm = struct.pack('<1003h', *(i-500 for i in range(1003)))
        self.audio = wav(self.pcm)
        point = dict(eligible=True, registered_grid=True, threshold=.5)
        self.calibration = dict(schema='complete-development-calibration-v1', target_id='hey_nyx',
            model=dict(sha256=complete.sha(self.model), bytes=len(self.model), target_id='hey_nyx'),
            candidate_promoted=False, acceptance_authorized=False, selected=point,
            threshold=.5, grid=[point.copy()],
            materials={phase: dict(role='new-development') for phase in ('synthetic', 'background')})
        self.seen = []

    def predictor(self, pcm):
        self.seen.append(pcm)
        frames = (len(pcm)//2+159)//160-2
        return bytes(i%256 for i in range((frames+2)//3)), dict(feature_frames=frames,
            model_sha256=complete.sha(self.model), frontend_sha256='constructed')

    def generate(self, predict=None):
        self.calibration_bytes = json.dumps(self.calibration).encode()
        return complete.generate(self.model, self.audio, self.calibration_bytes, predict or self.predictor)

    def log(self, report):
        events = ''.join(f'panel.wake.research: activation average=.9 step={step}\n'
                         for step in report['app_prime']['expected_device_event_steps'])
        return ('panel.wake.research: model='+report['model_sha256']+' threshold=.5\n'+events+
            f'panel.wake.research: complete selftest fixture={report["fixture_sha256"]} samples={report["samples"]} outputs={report["expected_outputs"]}\n'
            f'exact-PCM selftest outputs={report["expected_outputs"]} mismatches=0 passed=true\n')

    def test_preserves_source_tail_and_real_app_prime_with_fresh_whole_input_repeats(self):
        data, report = self.generate()
        expected_pcm = bytes(complete.PRIME_ROUNDED*2)+self.pcm
        self.assertEqual(self.seen, [expected_pcm, expected_pcm])
        fixture = complete.parse_fixture(data)
        self.assertEqual(fixture['pcm'], expected_pcm)
        self.assertEqual(fixture['samples'], 69163)
        self.assertEqual(fixture['outputs'], 144)
        self.assertEqual(report['final_wave_padding'], 117)
        self.assertEqual(report['final_feature_padding'], 1)
        self.assertEqual(report['app_prime']['source_samples'], 1003)
        self.assertEqual(report['app_prime']['tail_pad_samples'], 0)
        self.assertEqual(report['source_pcm_sha256'], complete.sha(self.pcm))
        result = audit(self.log(report), data, report, self.calibration_bytes)
        self.assertTrue(result['fixture_identity_in_uart'])
        self.assertEqual(result['outputs'], 144)

    def test_integer_event_latch_and_refractory(self):
        raw = bytes([0]*80+[255]*5+[0]*70+[255]*5)
        self.assertEqual(complete.event_steps(raw, .5), [83, 158])
        # A second high patch before67 steps does not cause another activation.
        self.assertEqual(complete.event_steps(bytes([0]*80+[255]*5+[0]*5+[255]*5), .5), [83])

    def test_source_format_truncation_and_size_fail_before_predicting(self):
        original = self.audio
        cases = [wav(self.pcm, rate=8000), wav(self.pcm, channels=2),
            wav(self.pcm, width=1), wav(bytes(959*2)), wav(bytes((256000-complete.PRIME_ROUNDED+1)*2)), original[:-1]]
        for audio in cases:
            with self.subTest(length=len(audio)):
                self.audio = audio
                with self.assertRaises((ValueError, wave.Error)):
                    self.generate()
                self.assertEqual(self.seen, [])

    def test_null_or_changed_operating_point_rejected_before_prediction(self):
        original = copy.deepcopy(self.calibration)
        cases = [{**original, 'selected': None}, {**original, 'threshold': .4},
            {**original, 'schema': 'legacy'}, {**original, 'target_id': 'hey_willow'},
            {**original, 'acceptance_authorized': True},
            {**original, 'model': {**original['model'], 'sha256': '0'*64}},
            {**original, 'selected': {**original['selected'], 'eligible': False}},
            {**original, 'selected': {**original['selected'], 'registered_grid': False}}]
        for calibration in cases:
            self.calibration = calibration
            with self.assertRaises(ValueError):
                self.generate()
            self.assertEqual(self.seen, [])

    def test_replay_count_model_metadata_or_uniform_output_mismatch_rejected(self):
        for mode in ['repeat', 'count', 'frames', 'model', 'uniform']:
            calls = 0

            def predict(pcm):
                nonlocal calls
                calls += 1
                raw, meta = self.predictor(pcm)
                if mode == 'repeat' and calls == 2:
                    raw = bytes([255])+raw[1:]
                if mode == 'count':
                    raw = raw[:-1]
                if mode == 'frames':
                    meta['feature_frames'] -= 1
                if mode == 'model':
                    meta['model_sha256'] = '0'*64
                if mode == 'uniform':
                    raw = bytes(len(raw))
                return raw, meta

            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.generate(predict)

    def test_binary_bounds_magic_and_model_identity(self):
        data, report = self.generate()
        for damaged in [data[:47], data[:-1], data+b'\0', b'NYXKWS01'+data[8:],
                data[:40]+struct.pack('<II', 0xffffffff, 1)+data[48:]]:
            with self.assertRaises(ValueError):
                complete.parse_fixture(damaged)
        changed = data[:8]+bytes(32)+data[40:]
        with self.assertRaises(ValueError):
            audit(self.log(report), changed, report, self.calibration_bytes)

    def test_uart_requires_exact_fixture_identity_before_verdict(self):
        data, report = self.generate()
        log = self.log(report)
        line = next(line for line in log.splitlines(True) if 'complete selftest fixture=' in line)
        cases = [log.replace(line, ''), log+line, log.replace(report['fixture_sha256'], 'f'*64),
            log.replace('samples=69163', 'samples=69164'), log.replace('mismatches=0', 'mismatches=1'),
            log.replace('passed=true', 'passed=false'), log.replace(line, '')+line]
        for changed in cases:
            with self.subTest(log=changed), self.assertRaises(ValueError):
                audit(changed, data, report, self.calibration_bytes)

    def test_report_hashes_padding_prime_and_raw_event_declarations(self):
        data, report = self.generate()
        for field, value in [('fixture_sha256', '0'*64), ('pcm_sha256', '0'*64),
                ('source_pcm_sha256', '0'*64), ('expected_outputs_sha256', '0'*64),
                ('final_wave_padding', 116), ('final_feature_padding', 0), ('tail_policy', 'truncate')]:
            changed = {**report, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit(self.log(report), data, changed, self.calibration_bytes)
        for field, value in [('host_event_steps', []), ('expected_device_event_steps', []),
                ('rounded_prime_samples', 68000), ('source_samples', 1004), ('threshold', .4)]:
            changed = copy.deepcopy(report)
            changed['app_prime'][field] = value
            with self.subTest(prime=field), self.assertRaises(ValueError):
                audit(self.log(report), data, changed, self.calibration_bytes)

    def test_predictor_backend_with_real_numpy_and_constructed_frontend_interpreter(self):
        import numpy as np
        input_pcm, instances = [], []

        def frontend(pcm):
            input_pcm.append(pcm.copy())
            frames = (len(pcm)+159)//160-2
            return (np.arange(frames*40).reshape(frames, 40)%500).astype(np.uint16)

        class Interpreter:
            def __init__(self, **kwargs):
                self.groups = []
                instances.append(self)

            def allocate_tensors(self):
                pass

            def get_input_details(self):
                return [dict(index=3, shape=[1, 3, 40], dtype=np.int8, quantization=(.125, -128))]

            def get_output_details(self):
                return [dict(index=7, shape=[1, 1], dtype=np.uint8, quantization=(1/256, 0))]

            def set_tensor(self, index, value):
                assert index == 3 and value.shape == (1, 3, 40) and value.dtype == np.int8
                self.groups.append(value.copy())

            def invoke(self):
                pass

            def get_tensor(self, index):
                assert index == 7
                return np.asarray([[int(self.groups[-1].astype(np.int64).sum())%256]], np.uint8)

        features = ModuleType('features')
        features.features = frontend
        inspect = ModuleType('prepare_runtime_selftest')
        inspect.inspect_model = lambda model, detector: dict(shape=detector.input['shape'], scale=detector.scale)
        interpreter = ModuleType('ai_edge_litert.interpreter')
        interpreter.Interpreter = Interpreter
        interpreter.OpResolverType = type('Resolver', (), {'BUILTIN_REF': 'constructed'})
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)/'model.tflite'
            model.write_bytes(self.model)
            with patch.dict('sys.modules', {'features': features, 'prepare_runtime_selftest': inspect,
                    'ai_edge_litert.interpreter': interpreter}), patch('importlib.metadata.version', return_value='constructed'):
                predictor = complete.make_predictor(model)
                data, report = self.generate(predictor)
                self.assertEqual(len(instances), 2)
                self.assertEqual(len(instances[0].groups), 144)
                np.testing.assert_array_equal(input_pcm[0], np.frombuffer(bytes(complete.PRIME_ROUNDED*2)+self.pcm, '<i2'))
                self.assertEqual(input_pcm[0].tobytes(), input_pcm[1].tobytes())
                # One final padded feature row, quantized zero=-128.
                np.testing.assert_array_equal(instances[0].groups[-1][0, -1], np.full(40, -128, np.int8))
                self.assertEqual(complete.parse_fixture(data)['outputs'], report['expected_outputs'])
                model.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Model changed'):
                    predictor(bytes(1920))


if __name__ == '__main__':
    unittest.main()
