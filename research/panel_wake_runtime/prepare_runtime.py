"""Bind an experimental firmware build to a separately calibrated INT8 model."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selftest", type=Path)
    args = parser.parse_args()
    data = args.model.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    calibration = json.loads(args.calibration.read_text())
    training_path = args.model.parent / "training.json"
    training = json.loads(training_path.read_text())
    if not 1024 <= len(data) <= 512 * 1024 or data[4:8] != b"TFL3":
        raise ValueError("Expected a bounded TFLite model")
    if digest != calibration["model_sha256"] or digest != training["model_sha256"]:
        raise ValueError("Model, training report and calibration identities differ")
    threshold = calibration["selected"]["threshold"]
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid frozen threshold")
    architecture = training["architecture"]
    flags = training["architecture_args"]
    if (architecture not in ("transfer60", "official64", "wide128")
            or "--stride" not in flags or flags[flags.index("--stride") + 1] != "3"):
        raise ValueError("Runtime supports only reviewed three-frame architectures")
    frames = training["input_frames"]
    if not isinstance(frames, int) or not 149 <= frames <= 335:
        raise ValueError("Unreviewed receptive field")
    selftest = None
    if args.selftest:
        selftest = args.selftest.read_bytes()
        if (len(selftest) < 48 or selftest[:8] != b"NYXKWS01"
                or selftest[8:40] != bytes.fromhex(digest)):
            raise ValueError("Self-test does not identify this model")
        samples, outputs = struct.unpack_from("<II", selftest, 40)
        if (not 960 <= samples <= 16000 * 8 or samples % 480
                or outputs != samples // 480 - 1 or len(selftest) != 48 + samples * 2 + outputs):
            raise ValueError("Invalid exact-PCM self-test layout")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hey_nyx_model.bin").write_bytes(data)
    if selftest:
        (args.output / "wake_selftest.bin").write_bytes(selftest)
    (args.output / "wake_config.h").write_text(
        "#pragma once\n"
        f'#define PANEL_WAKE_MODEL_SHA256 "{digest}"\n'
        f"#define PANEL_WAKE_THRESHOLD {threshold:.17g}f\n"
        f"#define PANEL_WAKE_PRIME_SAMPLES {(frames + 200) * 160}\n")
    (args.output / "manifest.json").write_text(json.dumps({
        "scope": "Experimental firmware build; recognition acceptance is not implied",
        "model_sha256": digest, "model_bytes": len(data),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "training_report_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(),
        "selftest_sha256": hashlib.sha256(selftest).hexdigest() if selftest else None,
        "threshold": threshold, "input_frames": frames, "sample_rate": 16000,
        "frontend_hop_samples": 160, "model_stride_frames": 3,
        "smoothing_steps": 5, "refractory_steps": 67,
        "validation_policy_operating_point": calibration.get("policy_operating_point"),
        "strict_validation_operating_point": calibration.get("viable_operating_point"),
        "prepare_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
