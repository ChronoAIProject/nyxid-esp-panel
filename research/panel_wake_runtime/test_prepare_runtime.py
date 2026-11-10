"""Check artifact binding/configuration boundaries without committing models."""
import hashlib
import json
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest


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


if __name__ == "__main__":
    unittest.main()
