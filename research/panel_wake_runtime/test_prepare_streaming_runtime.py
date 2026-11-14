"""Exercise packaging and actual CMake selection, without audio or a device."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import prepare_streaming_runtime as runtime


class StreamingProfile(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model.tflite'
        # Metadata fixture only. Real tensor allocation is checked by the board.
        self.model.write_bytes(bytes(4) + b'TFL3' + bytes(1016))
        self.profile = dict(schema=runtime.SCHEMA,
                            model_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest(),
                            context_frames=309, threshold=.91796875, smoothing_steps=5)
        self.profile_path = self.root / 'profile.json'
        self.output = self.root / 'generated'

    def prepare(self):
        self.profile_path.write_text(json.dumps(self.profile))
        return runtime.prepare(self.profile_path, self.model, self.output)

    def test_context309_packages_without_reports_or_expected_scores(self):
        manifest = self.prepare()
        self.assertEqual(manifest['actual_prime_samples'], 81600)
        self.assertEqual(manifest['requested_prime_samples'], 81440)
        self.assertEqual((self.output / 'hey_nyx_model.bin').read_bytes(), self.model.read_bytes())
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {'hey_nyx_model.bin', 'wake_config.h', 'manifest.json'})
        self.assertEqual(self.prepare(), manifest)
        # Reconfiguration must not silently replace a different model/profile.
        self.profile['threshold'] = .9
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_wrong_model_is_rejected_before_output(self):
        self.model.write_bytes(bytes(4) + b'TFL3' + bytes(1015) + b'X')
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_invalid_runtime_settings_are_rejected_before_output(self):
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

    def test_cmake_selects_streaming_only_in_isolated_development(self):
        self.profile_path.write_text(json.dumps(self.profile))
        component = Path(runtime.__file__).resolve().parent
        source = self.root / 'source'
        source.mkdir()
        (source / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.20)\nproject(streaming_configure LANGUAGES CXX)\n'
            'macro(idf_component_register)\n'
            f'add_library(runtime STATIC "{component}/panel_wake_runtime.cpp")\n'
            'set(COMPONENT_LIB runtime)\nendmacro()\n'
            'function(target_add_binary_data target path kind)\n'
            'if(NOT EXISTS "${path}")\nmessage(FATAL_ERROR "Missing model")\nendif()\n'
            'file(APPEND "${CMAKE_BINARY_DIR}/embedded.txt" "${path}\\n")\nendfunction()\n'
            f'add_subdirectory("{component}" runtime)\n'
            'get_target_property(defs runtime COMPILE_DEFINITIONS)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/definitions.txt" "${defs}")\n')
        cases = [([], False), (['-DNYX_PANEL_NODE_DEV_MODE=ON'], True)]
        for option in ('CALIBRATION', 'SELFTEST', 'CANDIDATE_PROFILE', 'COMPLETE_PROFILE', 'EMBEDDING_BUNDLE'):
            name = ('NYX_PANEL_EXPERIMENTAL_' + option if option == 'EMBEDDING_BUNDLE'
                    else 'NYX_PANEL_EXPERIMENTAL_WAKE_' + option)
            cases.append((['-DNYX_PANEL_NODE_DEV_MODE=ON', '-D' + name + '=legacy'], False))
        for index, (extra, succeeds) in enumerate(cases):
            with self.subTest(extra=extra):
                build = self.root / f'build{index}'
                result = subprocess.run(['cmake', '-S', str(source), '-B', str(build),
                    '-DPYTHON=' + sys.executable,
                    '-DNYX_PANEL_EXPERIMENTAL_WAKE_MODEL=' + str(self.model),
                    '-DNYX_PANEL_EXPERIMENTAL_WAKE_STREAMING_PROFILE=' + str(self.profile_path),
                    *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)
                if succeeds:
                    self.assertEqual((build / 'definitions.txt').read_text(), 'PANEL_WAKE_STREAMING_PROFILE=1')
                    self.assertEqual(len((build / 'embedded.txt').read_text().splitlines()), 1)
                    manifest = json.loads((build / 'runtime/generated/manifest.json').read_text())
                    self.assertEqual(manifest['actual_prime_samples'], 81600)


if __name__ == '__main__':
    unittest.main()
