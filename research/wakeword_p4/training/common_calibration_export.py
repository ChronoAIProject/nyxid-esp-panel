"""Freeze shared training-only representative inputs, then export without retraining.

These auxiliary INT8 exports isolate representative input differences. They do not
overwrite original models, select thresholds, or establish independent acceptance.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--features", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export")
    export.add_argument("--model-dir", type=Path, required=True)
    export.add_argument("--representatives", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; preserve all earlier exports")

    if args.mode == "prepare":
        index = args.features / "features.json"
        rows = json.loads(index.read_text())
        groups = defaultdict(list)
        for row in rows:
            if row["split"] == "training" and row["variant"] == 0 and row["frames"] >= 150:
                groups[(row["engine"], row["label"])].append(row)
        if not groups or {label for _, label in groups} != {0, 1}:
            raise ValueError("Require training positive and confusing-negative sources")
        raw = np.memmap(args.features / "features.bin", mode="r", dtype="<u2").reshape(-1, 40)
        rng = np.random.default_rng(57726)
        selected, tensors = [], []
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda r: (r["path"], r["offset"]))
            if len(group) < 20:
                raise ValueError("Require 20 distinct training rows per available engine/label group")
            for i in rng.choice(len(group), 20, replace=False):
                row = group[int(i)]
                start = int(rng.integers(row["frames"] - 150 + 1))
                x = raw[row["offset"] + start:row["offset"] + start + 150]
                if x.shape != (150, 40):
                    raise ValueError("Feature crop outside binary")
                tensors.append(np.clip(x.astype(np.float32) / 25.6, 0, 26))
                selected.append({"row": row, "crop_start_frame": start})
        values = np.stack(tensors)
        args.output.mkdir(parents=True)
        np.save(args.output / "inputs.npy", values, allow_pickle=False)
        report = {"role": "Frozen shared training-only INT8 representative inputs",
                  "features_index_sha256": sha(index), "inputs_sha256": sha(args.output / "inputs.npy"),
                  "seed": 57726, "frames": 150, "groups": [list(k) for k in sorted(groups)],
                  "windows": len(values), "rows": selected,
                  "policy": "20 distinct variant0 training rows per available engine/label; uniform crop; no augmentation, prefix replacement, validation, final data, or model-score selection",
                  "limits": "No separate ambient rows; diagnostic calibration choice, not a demonstrated optimal deployment calibration"}
    else:
        rep_path = args.representatives / "report.json"
        rep = json.loads(rep_path.read_text())
        inputs = args.representatives / "inputs.npy"
        if sha(inputs) != rep["inputs_sha256"] or any(r["row"]["split"] != "training" for r in rep["rows"]):
            raise ValueError("Representative hash/split mismatch")
        values = np.load(inputs, allow_pickle=False)
        if values.shape != (rep["windows"], 150, 40) or not np.isfinite(values).all():
            raise ValueError("Invalid representative tensors")
        training_path = args.model_dir / "training.json"
        training = json.loads(training_path.read_text())
        if training["context_frames"] != 150:
            raise ValueError("This protocol is frozen for context150 models")
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(2)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(args.model_dir / "saved_model"))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter._experimental_variable_quantization = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.uint8

        def representative():
            yield [np.zeros((1, 3, 40), dtype=np.float32)]
            yield [np.full((1, 3, 40), 26, dtype=np.float32)]
            for window in values:
                for at in range(0, 150, 3):
                    yield [window[None, at:at+3]]

        converter.representative_dataset = representative
        data = converter.convert()
        args.output.mkdir(parents=True)
        (args.output / "hey_nyx.tflite").write_bytes(data)
        report = {"role": "Auxiliary matched-representative INT8 export; not final acceptance",
                  "training_report_sha256": sha(training_path),
                  "original_model_sha256": sha(args.model_dir / "hey_nyx.tflite"),
                  "representative_manifest_sha256": sha(rep_path),
                  "representative_inputs_sha256": sha(inputs),
                  "saved_model_files_sha256": {str(p.relative_to(args.model_dir)): sha(p) for p in sorted((args.model_dir / "saved_model").rglob("*")) if p.is_file()},
                  "model_sha256": hashlib.sha256(data).hexdigest(), "model_bytes": len(data),
                  "tensorflow": tf.__version__, "training_or_threshold_selection_performed": False}
    report["source_sha256"] = sha(Path(__file__))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "saved_model_files_sha256")}, indent=2))


if __name__ == "__main__":
    main()
