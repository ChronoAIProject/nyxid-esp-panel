"""Diagnose selected validation negative texts at an already frozen threshold.

Does not calibrate, train, read testing audio, or turn ASR guesses into labels.
Useful for checking whether a proposed vocabulary gap actually causes errors.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import Detector, smooth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-prefix", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh report path")
    calibration = json.loads(args.calibration.read_text())
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if calibration.get("model_sha256") != digest:
        raise ValueError("Calibration does not identify this frozen model")
    index = args.features/"features.json"
    rows = [r for r in json.loads(index.read_text()) if r["split"] == "validation" and r["label"] == 0
            and any(r["path"].startswith(prefix+"/") for prefix in args.source_prefix)]
    if not rows:
        raise ValueError("No selected validation negative vocabulary")
    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    detector = Detector(args.model)
    threshold = calibration["selected"]["threshold"]
    groups = defaultdict(list)
    for row in rows:
        detector.reset()
        values = smooth(detector.predict(raw[row["offset"]:row["offset"]+row["frames"]]))
        groups[(row["engine"], row["text"])].append(float(values.max()) if len(values) else 0.)
    result = {"model_sha256": digest, "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "feature_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
              "threshold": threshold, "source_prefixes": args.source_prefix, "split": "validation",
              "clip_variants": len(rows), "false_clips": sum(v >= threshold for values in groups.values() for v in values),
              "by_engine_and_text": [{"engine": engine, "text": text, "variants": len(values),
                                      "false_clips": sum(v >= threshold for v in values),
                                      "max_score": max(values)} for (engine, text), values in sorted(groups.items())],
              "scope": "Validation vocabulary diagnostic only. Correlated synthetic variants; frozen threshold, no calibration or human acceptance."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("threshold", "clip_variants", "false_clips")}), flush=True)


if __name__ == "__main__":
    main()
