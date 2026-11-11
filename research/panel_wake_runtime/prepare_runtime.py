"""Bind an experimental firmware build to a separately calibrated INT8 model."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


GPU_PROFILE = Path(__file__).parent / "plans/gpu-v30-runtime-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_gpu_reports(profile, training_bytes, calibration_bytes):
    if (sha(training_bytes) != profile["training_report_sha256"]
            or sha(calibration_bytes) != profile["calibration_sha256"]):
        raise ValueError("Expected the exact frozen GPU training and calibration reports")
    training, calibration = json.loads(training_bytes), json.loads(calibration_bytes)
    schema = profile.get("calibration_schema", "expanded-english-v1")
    if schema == "expanded-english-v1":
        calibration_matches = (
            calibration["selected"]["english"]["threshold"] == profile["threshold"]
            and calibration["independent_final_scored"] is False)
    elif schema == "joint-development-v1":
        # Eligibility is computed by the existing external calibration stage.
        # This packager authenticates its frozen selected result; it does not
        # reproduce the grid, select a threshold or confer acceptance.
        selected = calibration.get("selected")
        calibration_matches = (
            training.get("steps") == 120000
            and isinstance(selected, dict) and selected.get("eligible") is True
            and selected.get("threshold") == profile["threshold"]
            and calibration.get("final_data_used") is False)
    else:
        raise ValueError("Unreviewed GPU calibration schema")
    if (any(training[key] != profile[key] for key in
            ("model_sha256", "model_bytes", "architecture", "architecture_args",
             "selected_step", "validation_window_phase"))
            or training["context_frames"] != profile["input_frames"]
            or calibration["model_sha256"] != profile["model_sha256"]
            or calibration["threshold"] != profile["threshold"]
            or calibration["sliding_window"] != profile["smoothing_steps"]
            or calibration["validation_only"] is not True
            or calibration["viable_operating_point"] is not True
            or not calibration_matches or profile["refractory_steps"] != 67):
        raise ValueError("Frozen GPU topology, calibration or event policy differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selftest", type=Path)
    parser.add_argument("--gpu-profile", type=Path,
                        help="Explicit frozen teacher60 candidate profile; default pins original v30")
    args = parser.parse_args()
    data = args.model.read_bytes()
    digest = sha(data)
    calibration_bytes = args.calibration.read_bytes()
    calibration = json.loads(calibration_bytes)
    training_path = args.model.parent / "training.json"
    training_bytes = training_path.read_bytes()
    training = json.loads(training_bytes)
    if not 1024 <= len(data) <= 512 * 1024 or data[4:8] != b"TFL3":
        raise ValueError("Expected a bounded TFLite model")
    if digest != calibration["model_sha256"] or digest != training["model_sha256"]:
        raise ValueError("Model, training report and calibration identities differ")
    architecture = training["architecture"]
    reference_profile = json.loads(GPU_PROFILE.read_bytes())
    profile_bytes = (args.gpu_profile or GPU_PROFILE).read_bytes()
    profile = json.loads(profile_bytes)
    if args.gpu_profile and (
            architecture != "teacher60" or profile["architecture"] != reference_profile["architecture"]
            or profile["architecture_args"] != reference_profile["architecture_args"]
            or profile["input_frames"] != reference_profile["input_frames"]
            or profile["validation_window_phase"] != reference_profile["validation_window_phase"]
            or profile["smoothing_steps"] != reference_profile["smoothing_steps"]
            or profile.get("accepted") is not False):
        raise ValueError("Explicit GPU profile must retain the reviewed research topology and event policy")
    gpu = architecture == profile["architecture"] or digest == profile["model_sha256"]
    if gpu:
        if digest != profile["model_sha256"] or len(data) != profile["model_bytes"]:
            raise ValueError("Expected the exact frozen GPU model identified by its profile")
        validate_gpu_reports(profile, training_bytes, calibration_bytes)
        if not args.selftest:
            raise ValueError("GPU development build requires an exact-PCM self-test")
        threshold, smoothing_steps, frames = (profile[key] for key in ("threshold", "smoothing_steps", "input_frames"))
    else:
        flags = training["architecture_args"]
        stride = flags.index("--stride") if "--stride" in flags else -1
        if (architecture not in ("transfer60", "official64", "wide128")
                or stride < 0 or flags[stride + 1:stride + 2] != ["3"]):
            raise ValueError("Runtime supports only reviewed three-frame architectures")
        threshold = calibration["selected"]["threshold"]
        smoothing_steps = calibration.get("smoothing_steps", 5)
        frames = training["input_frames"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid frozen threshold")
    if type(smoothing_steps) is not int or smoothing_steps not in (3, 5, 7):
        raise ValueError("Unsupported calibration smoothing_steps")
    if type(frames) is not int or not 149 <= frames <= 335:
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
        expected = selftest[48 + samples * 2:]
        if gpu and (outputs < (frames + 2) // 3 + smoothing_steps or min(expected) == max(expected)):
            raise ValueError("GPU self-test must cover complete model history and varied outputs")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hey_nyx_model.bin").write_bytes(data)
    if selftest:
        (args.output / "wake_selftest.bin").write_bytes(selftest)
    (args.output / "wake_config.h").write_text(
        "#pragma once\n"
        f'#define PANEL_WAKE_MODEL_SHA256 "{digest}"\n'
        f"#define PANEL_WAKE_THRESHOLD {threshold:.17g}f\n"
        f"#define PANEL_WAKE_SMOOTHING_STEPS {smoothing_steps}\n"
        f"#define PANEL_WAKE_PRIME_SAMPLES {(frames + 200) * 160}\n")
    (args.output / "manifest.json").write_text(json.dumps({
        "scope": "Experimental firmware build; recognition acceptance is not implied",
        "model_sha256": digest, "model_bytes": len(data),
        "calibration_sha256": sha(calibration_bytes),
        "training_report_sha256": sha(training_bytes),
        "selftest_sha256": sha(selftest) if selftest else None,
        "architecture": architecture,
        "threshold": threshold, "input_frames": frames, "sample_rate": 16000,
        "frontend_hop_samples": 160, "model_stride_frames": 3,
        "smoothing_steps": smoothing_steps, "refractory_steps": 67,
        "validation_policy_operating_point": calibration.get("policy_operating_point"),
        "strict_validation_operating_point": calibration.get("viable_operating_point"),
        "prepare_source_sha256": sha(Path(__file__).read_bytes()),
        **({"gpu_profile_sha256": sha(profile_bytes), "gpu_source_commit": profile["source_commit"],
            "validation_window_phase": profile["validation_window_phase"]} if gpu else {})
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
