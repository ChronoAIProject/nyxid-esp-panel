"""Exercise fixture event arithmetic with injected analytic score bytes only."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import prepare_runtime_selftest as selftest


class NumericDetector:
    output_scale = 1 / 256
    output_zero = 0

    def __init__(self, _path):
        pass

    def reset(self):
        pass

    def predict(self, count):
        raw = [0] * 10 + [253, 253, 227, 231, 211]
        return np.asarray(raw + [0] * (count - len(raw)), np.float32) / 256


class FixtureArithmetic(unittest.TestCase):
    def test_explicit_mean_matches_device_and_preserves_fixture_pcm_and_score_bytes(self):
        # Detector/frontend are injected; this is not a model or speech test.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.tflite'
            model.write_bytes(b'analytic metadata fixture only')
            digest = hashlib.sha256(model.read_bytes()).hexdigest()
            (root / 'training.json').write_text(json.dumps(dict(
                model_sha256=digest, architecture='teacher60', context_frames=150)))
            audio = root / 'audio.fixture'
            audio.write_bytes(b'no audio; sf.read is injected with zero samples')
            fixtures = []
            for name, fields in [('legacy', {}), ('canonical', dict(event_arithmetic=selftest.UINT8_BINARY64))]:
                calibration = root / (name + '-calibration.json')
                calibration.write_text(json.dumps(dict(model_sha256=digest, threshold=.91796875,
                    sliding_window=5, **fields)))
                output = root / (name + '.bin')
                argv = ['prepare_runtime_selftest.py', '--model', str(model), '--audio', str(audio),
                    '--output', str(output), '--app-prime-calibration', str(calibration)]
                with patch.object(sys, 'argv', argv), patch.object(selftest, 'Detector', NumericDetector), \
                     patch.object(selftest, 'inspect_model', return_value={'analytic_fixture': True}), \
                     patch.object(selftest, 'features', side_effect=lambda pcm: len(pcm) // 480 - 1), \
                     patch.object(selftest.sf, 'read', return_value=(np.zeros(960, np.int16), 16000)):
                    selftest.main()
                report = json.loads(output.with_suffix('.json').read_text())
                fixtures.append(output.read_bytes())
                self.assertEqual(report['app_prime']['expected_device_event_steps'], [15])
                expected_host = [] if name == 'legacy' else [15]
                self.assertEqual(report['app_prime']['host_event_steps'], expected_host)
            self.assertEqual(fixtures[0], fixtures[1])


if __name__ == '__main__':
    unittest.main()
