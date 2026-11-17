"""Validate the shipping C161 artifact, bounded settings and build selection."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / 'components/panel_wake_runtime'
spec = importlib.util.spec_from_file_location('streaming_wake', COMPONENT / 'prepare_streaming_runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class StreamingWake(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model.tflite'
        self.model.write_bytes((ROOT / 'models/hey-nyx-c161/model.tflite').read_bytes())
        self.profile = json.loads((ROOT / 'models/hey-nyx-c161/profile.json').read_text())
        self.profile_path = self.root / 'profile.json'
        self.output = self.root / 'generated'

    def prepare(self):
        self.profile_path.write_text(json.dumps(self.profile))
        return runtime.prepare(self.profile_path, self.model, self.output)

    def test_shipping_model_identity_and_deployed_settings(self):
        self.assertEqual(hashlib.sha256(self.model.read_bytes()).hexdigest(),
                         'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81')
        manifest = self.prepare()
        self.assertEqual((manifest['sample_rate'], manifest['context_frames'],
                          manifest['threshold'], manifest['smoothing_steps'],
                          manifest['refractory_steps'], manifest['actual_prime_samples']),
                         (16000, 309, .91796875, 5, 67, 81600))
        self.assertEqual((self.output / 'hey_nyx_model.bin').read_bytes(), self.model.read_bytes())
        self.assertEqual(self.prepare(), manifest)
        self.profile['threshold'] = .9
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_wrong_model_is_rejected_before_output(self):
        self.model.write_bytes(self.model.read_bytes()[:-1] + b'X')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_settings_are_rejected_before_output(self):
        original = self.profile.copy()
        for key, value in [('schema', 'unknown'), ('context_frames', True),
                           ('context_frames', 0), ('context_frames', 1001),
                           ('threshold', float('nan')), ('threshold', 0),
                           ('threshold', 1), ('smoothing_steps', 0)]:
            with self.subTest(key=key, value=value):
                self.profile = {**original, key: value}
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_engine_selection_is_independent_of_hardware_and_server(self):
        source = self.root / 'select.cmake'
        source.write_text(f'include("{ROOT}/cmake/panel_wake.cmake")\n'
                          'file(WRITE "${OUTPUT}" "${NYX_PANEL_WAKE_ENGINE}")\n')
        for index, (engine, expected, succeeds) in enumerate([
            (None, 'hey-nyx', True), ('hey-nyx', 'hey-nyx', True),
            ('hi-esp', 'hi-esp', True), ('unknown', '', False)]):
            with self.subTest(engine=engine):
                output = self.root / f'engine{index}'
                args = ['cmake', f'-DOUTPUT={output}']
                if engine is not None:
                    args += [f'-DNYX_PANEL_WAKE_ENGINE={engine}']
                result = subprocess.run(args + ['-P', str(source)], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
                if succeeds:
                    self.assertEqual(output.read_text(), expected)
