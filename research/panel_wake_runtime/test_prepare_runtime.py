"""Check artifact binding/configuration boundaries without committing models."""
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_runtime


class PrepareRuntime(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wake-runtime-config-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Only a header-shaped fixture: this test checks the build metadata
        # boundary. AllocateTensors and board probes validate actual operators.
        self.model = self.root / "hey_nyx.tflite"
        self.model.write_bytes(b"\0\0\0\0TFL3" + bytes(1016))
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.training = {"model_sha256": self.digest, "architecture": "transfer60",
                         "architecture_args": ["--stride", "3"], "input_frames": 335}
        self.calibration = {"model_sha256": self.digest,
                            "selected": {"threshold": .1}, "viable_operating_point": False}

    def run_prepare(self, selftest=None):
        (self.root / "training.json").write_text(json.dumps(self.training))
        calibration = self.root / "threshold.json"
        calibration.write_text(json.dumps(self.calibration))
        command = [sys.executable, str(Path(__file__).with_name("prepare_runtime.py")),
            "--model", str(self.model), "--calibration", str(calibration),
            "--output", str(self.root / "output")]
        if selftest is not None:
            fixture = self.root / "selftest.bin"
            fixture.write_bytes(selftest)
            command += ["--selftest", str(fixture)]
        return subprocess.run(command, text=True, capture_output=True)

    def assert_rejected(self):
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "output").exists())

    def test_binds_model_and_frozen_threshold_without_acceptance_claim(self):
        self.assertEqual(self.run_prepare().returncode, 0)
        output = self.root / "output"
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["model_sha256"], self.digest)
        self.assertEqual(manifest["threshold"], .1)
        self.assertEqual(manifest["smoothing_steps"], 5)
        self.assertFalse(manifest["strict_validation_operating_point"])
        self.assertIn("acceptance is not implied", manifest["scope"])
        self.assertEqual((output / "hey_nyx_model.bin").read_bytes(), self.model.read_bytes())
        self.assertIn("85600", (output / "wake_config.h").read_text())

    def test_calibration_for_different_model_rejected(self):
        self.calibration["model_sha256"] = "0" * 64
        self.assert_rejected()

    def test_calibration_binds_reviewed_window_to_runtime_and_manifest(self):
        self.calibration["smoothing_steps"] = 7
        self.assertEqual(self.run_prepare().returncode, 0)
        output = self.root / "output"
        self.assertEqual(json.loads((output / "manifest.json").read_text())["smoothing_steps"], 7)
        self.assertIn("#define PANEL_WAKE_SMOOTHING_STEPS 7", (output / "wake_config.h").read_text())

    def test_invalid_explicit_smoothing_rejected_before_output(self):
        for steps in (None, 0, 4, 8, 5., "5", True, [5]):
            with self.subTest(steps=steps):
                self.calibration["smoothing_steps"] = steps
                self.assert_rejected()

    def test_training_report_for_different_model_rejected(self):
        self.training["model_sha256"] = "0" * 64
        self.assert_rejected()

    def test_invalid_thresholds_rejected(self):
        for threshold in (0, 1, -.1, float("nan"), float("inf"), "0.1", True):
            with self.subTest(threshold=threshold):
                self.calibration["selected"]["threshold"] = threshold
                self.assert_rejected()

    def test_unreviewed_architecture_stride_context_rejected(self):
        original = self.training.copy()
        for field, value in (("architecture", "unknown"), ("architecture_args", ["--stride", "2"]),
                             ("input_frames", 148), ("input_frames", 336)):
            with self.subTest(field=field, value=value):
                self.training = {**original, field: value}
                self.assert_rejected()

    def test_truncated_model_rejected(self):
        self.model.write_bytes(b"\0\0\0\0TFL3")
        self.digest = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.training["model_sha256"] = self.digest
        self.calibration["model_sha256"] = self.digest
        self.assert_rejected()

    def test_selftest_bound_to_model_and_copied_exactly(self):
        data = b"NYXKWS01" + bytes.fromhex(self.digest) + struct.pack("<II", 960, 1) + bytes(1921)
        self.assertEqual(self.run_prepare(data).returncode, 0)
        output = self.root / "output"
        self.assertEqual((output / "wake_selftest.bin").read_bytes(), data)
        self.assertEqual(json.loads((output / "manifest.json").read_text())["selftest_sha256"],
                         hashlib.sha256(data).hexdigest())

    def test_wrong_model_or_corrupt_selftest_rejected_before_output(self):
        data = b"NYXKWS01" + bytes.fromhex(self.digest) + struct.pack("<II", 960, 1) + bytes(1921)
        fixtures = [data[:8] + bytes(32) + data[40:], data[:-1],
                    data[:40] + struct.pack("<II", 959, 1) + data[48:],
                    data[:40] + struct.pack("<II", 960, 2) + data[48:], data[:47]]
        for fixture in fixtures:
            with self.subTest(size=len(fixture)):
                self.assertNotEqual(self.run_prepare(fixture).returncode, 0)
                self.assertFalse((self.root / "output").exists())

    def test_unreviewed_teacher60_rejected_with_consistent_generic_reports(self):
        self.training.update(architecture="teacher60", input_frames=150)
        self.assert_rejected()

    def test_gpu_profile_requires_exact_reports_and_informative_fixture(self):
        # Exercise file generation only. These fixture bytes aren't an actual
        # model, and the production profile is never substituted in a build.
        profile = json.loads(prepare_runtime.GPU_PROFILE.read_bytes())
        profile.update(model_sha256=self.digest, model_bytes=self.model.stat().st_size)
        fields = ("model_sha256", "model_bytes", "architecture", "architecture_args",
                  "selected_step", "validation_window_phase")
        training = {key: profile[key] for key in fields} | {"context_frames": 150}
        calibration = {"model_sha256": self.digest, "threshold": .91796875,
                       "selected": {"english": {"threshold": .91796875}},
                       "sliding_window": 5, "validation_only": True,
                       "viable_operating_point": True, "independent_final_scored": False}
        training_path = self.root / "training.json"
        calibration_path = self.root / "calibration.json"
        for path, data, key in ((training_path, training, "training_report_sha256"),
                                (calibration_path, calibration, "calibration_sha256")):
            path.write_text(json.dumps(data))
            profile[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        profile_path = self.root / "profile.json"
        profile_path.write_text(json.dumps(profile))
        output = self.root / "output"
        argv = ["prepare_runtime.py", "--model", str(self.model), "--calibration", str(calibration_path),
                "--output", str(output)]
        fixture_path = self.root / "fixture.bin"
        outputs = 55
        samples = (outputs + 1) * 480
        fixture = (b"NYXKWS01" + bytes.fromhex(self.digest) + struct.pack("<II", samples, outputs)
                   + bytes(samples * 2) + bytes(range(outputs)))
        with patch.object(prepare_runtime, "GPU_PROFILE", profile_path):
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "requires an exact-PCM"):
                prepare_runtime.main()
            self.assertFalse(output.exists())
            argv += ["--selftest", str(fixture_path)]
            with patch.object(sys, "argv", argv):
                fixture_path.write_bytes(fixture[:-outputs] + bytes(outputs))
                with self.assertRaisesRegex(ValueError, "varied outputs"):
                    prepare_runtime.main()
                self.assertFalse(output.exists())
                fixture_path.write_bytes(fixture)
                calibration_path.write_text(json.dumps(calibration) + "\n")
                with self.assertRaisesRegex(ValueError, "exact frozen GPU"):
                    prepare_runtime.main()
                self.assertFalse(output.exists())
                calibration_path.write_text(json.dumps(calibration))
                prepare_runtime.main()
        manifest = json.loads((output / "manifest.json").read_bytes())
        self.assertEqual(manifest["architecture"], "teacher60")
        self.assertEqual(manifest["threshold"], .91796875)
        self.assertEqual(manifest["input_frames"], 150)
        self.assertEqual(manifest["gpu_source_commit"], profile["source_commit"])
        self.assertEqual((output / "wake_selftest.bin").read_bytes(), fixture)

    def test_explicit_candidate_authenticates_selected_reports_and_selftest(self):
        # This is a metadata boundary fixture, never a deployable model/profile.
        profile = json.loads(prepare_runtime.GPU_PROFILE.read_bytes())
        profile.update(model_sha256=self.digest, model_bytes=self.model.stat().st_size,
                       calibration_schema="joint-development-v1", selected_step=45000)
        fields = ("model_sha256", "model_bytes", "architecture", "architecture_args",
                  "selected_step", "validation_window_phase")
        training = {key: profile[key] for key in fields} | {"context_frames": 150, "steps": 120000}
        calibration = {"model_sha256": self.digest, "threshold": profile['threshold'],
                       "selected": {"threshold": profile['threshold'], "eligible": True},
                       "sliding_window": 5, "validation_only": True,
                       "viable_operating_point": True, "final_data_used": False}
        training_path = self.root / 'training.json'
        calibration_path = self.root / 'calibration.json'
        profile_path = self.root / 'candidate.json'
        fixture_path = self.root / 'fixture.bin'
        outputs = 55
        samples = (outputs+1)*480
        fixture_path.write_bytes(b'NYXKWS01'+bytes.fromhex(self.digest)
                                 + struct.pack('<II', samples, outputs)+bytes(samples*2)+bytes(range(outputs)))
        output = self.root / 'output'
        argv = ['prepare_runtime.py', '--model', str(self.model), '--calibration', str(calibration_path),
                '--gpu-profile', str(profile_path), '--selftest', str(fixture_path), '--output', str(output)]

        def write_metadata(candidate, calibrated):
            training_path.write_text(json.dumps(training))
            calibration_path.write_text(json.dumps(calibrated))
            candidate = {**candidate,
                         'training_report_sha256': hashlib.sha256(training_path.read_bytes()).hexdigest(),
                         'calibration_sha256': hashlib.sha256(calibration_path.read_bytes()).hexdigest()}
            profile_path.write_text(json.dumps(candidate))

        for name, candidate, calibrated in [
                ('rejected calibration', profile, {**calibration, 'viable_operating_point': False}),
                ('no selected point', profile, {**calibration, 'selected': None}),
                ('selected threshold differs', profile, {**calibration, 'selected': {'threshold': .5, 'eligible': True}}),
                ('unreviewed topology', {**profile, 'architecture_args': ['--stride', '2']}, calibration),
                ('unknown schema', {**profile, 'calibration_schema': 'unknown'}, calibration),
                ('acceptance claimed', {**profile, 'accepted': True}, calibration)]:
            with self.subTest(name=name):
                write_metadata(candidate, calibrated)
                with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                    prepare_runtime.main()
                self.assertFalse(output.exists())
        write_metadata(profile, calibration)
        selftest_at = argv.index('--selftest')
        without_selftest = argv[:selftest_at] + argv[selftest_at+2:]
        with patch.object(sys, 'argv', argv):
            # An explicit candidate still cannot omit a model-specific self-test.
            with patch.object(sys, 'argv', without_selftest), self.assertRaises(ValueError):
                prepare_runtime.main()
            self.assertFalse(output.exists())
            prepare_runtime.main()
        manifest = json.loads((output/'manifest.json').read_bytes())
        self.assertEqual(manifest['gpu_profile_sha256'], hashlib.sha256(profile_path.read_bytes()).hexdigest())
        self.assertEqual(manifest['training_report_sha256'], hashlib.sha256(training_path.read_bytes()).hexdigest())
        self.assertEqual(manifest['threshold'], profile['threshold'])
        self.assertEqual((output/'wake_selftest.bin').read_bytes(), fixture_path.read_bytes())

    def test_explicit_arithmetic_binds_both_reports_and_compiled_threshold(self):
        # Metadata fixture only, not a deployable model or eligible candidate.
        profile = json.loads(prepare_runtime.GPU_PROFILE.read_bytes())
        profile.update(model_sha256=self.digest, model_bytes=self.model.stat().st_size,
                       calibration_schema='joint-development-v1', threshold=.8,
                       event_arithmetic=prepare_runtime.UINT8_BINARY64)
        training = {key: profile[key] for key in ('model_sha256', 'model_bytes', 'architecture',
            'architecture_args', 'selected_step', 'validation_window_phase')}
        training.update(context_frames=150, steps=120000)
        calibration = dict(model_sha256=self.digest, threshold=.8, selected=dict(threshold=.8, eligible=True),
            sliding_window=5, validation_only=True, viable_operating_point=True, final_data_used=False,
            event_arithmetic=prepare_runtime.UINT8_BINARY64)
        training_bytes = json.dumps(training).encode()
        calibration_path = self.root / 'calibration.json'
        profile_path = self.root / 'profile.json'
        (self.root / 'training.json').write_bytes(training_bytes)
        samples = 56 * 480
        fixture_path = self.root / 'fixture.bin'
        fixture_path.write_bytes(b'NYXKWS01' + bytes.fromhex(self.digest)
            + struct.pack('<II', samples, 55) + bytes(samples * 2) + bytes(range(55)))
        output = self.root / 'output'
        argv = ['prepare_runtime.py', '--model', str(self.model), '--calibration', str(calibration_path),
                '--gpu-profile', str(profile_path), '--selftest', str(fixture_path), '--output', str(output)]

        def write(candidate, calibrated):
            blob = json.dumps(calibrated).encode()
            calibration_path.write_bytes(blob)
            profile_path.write_text(json.dumps({**candidate,
                'training_report_sha256': hashlib.sha256(training_bytes).hexdigest(),
                'calibration_sha256': hashlib.sha256(blob).hexdigest()}))

        missing_profile = {key: value for key, value in profile.items() if key != 'event_arithmetic'}
        missing_calibration = {key: value for key, value in calibration.items() if key != 'event_arithmetic'}
        for name, candidate, calibrated in [
                ('missing profile marker', missing_profile, calibration),
                ('missing calibration marker', profile, missing_calibration),
                ('unknown arithmetic', {**profile, 'event_arithmetic': 'unknown'}, calibration),
                ('explicit null arithmetic', {**profile, 'event_arithmetic': None}, calibration),
                ('unknown calibration arithmetic', profile, {**calibration, 'event_arithmetic': 'unknown'}),
                ('null calibration arithmetic', profile, {**calibration, 'event_arithmetic': None}),
                ('legacy schema', {**profile, 'calibration_schema': 'expanded-english-v1'}, calibration),
                ('changed smoothing', {**profile, 'smoothing_steps': 7}, calibration)]:
            with self.subTest(name=name):
                write(candidate, calibrated)
                with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                    prepare_runtime.main()
                self.assertFalse(output.exists())
        write(profile, calibration)
        with patch.object(sys, 'argv', argv):
            prepare_runtime.main()
        manifest = json.loads((output / 'manifest.json').read_text())
        self.assertEqual(manifest['event_arithmetic'], prepare_runtime.UINT8_BINARY64)
        program = self.root / 'threshold.cpp'
        program.write_text('#include "wake_config.h"\n#include <type_traits>\n'
            'static_assert(std::is_same_v<decltype(PANEL_WAKE_THRESHOLD), double>);\n'
            'int main() { const double score = 1024.0 / 1280.0; '
            'return score >= PANEL_WAKE_THRESHOLD ? 0 : 1; }\n')
        subprocess.run(['c++', '-std=c++17', '-I', str(output), str(program),
                        '-o', str(self.root / 'threshold')], check=True, capture_output=True)
        subprocess.run([str(self.root / 'threshold')], check=True)


if __name__ == "__main__":
    unittest.main()
