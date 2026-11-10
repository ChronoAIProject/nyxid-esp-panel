"""Measure a validation-frozen Hey Nyx model on longer continuous negatives."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import chi2

from evaluate import Detector, events, smooth
from detector_policy import calibrated_smoothing_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="Optional completed Hi ESP report for identical PCM")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    calibration = json.loads(args.calibration.read_text())
    smoothing_steps = calibrated_smoothing_steps(calibration)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if calibration.get("model_sha256") != model_hash:
        raise ValueError("Calibration does not identify this frozen model")
    corpus_path = args.corpus/"manifest.json"
    corpus = json.loads(corpus_path.read_text())
    args.output.mkdir(parents=True)
    detector = Detector(args.model)
    threshold = calibration["selected"]["threshold"]
    rows = []
    started = time.monotonic()
    for index, record in enumerate(corpus["records"]):
        path = args.corpus/record["path"]
        data = np.load(path, mmap_mode="r")
        if data.shape != (record["frames"], 40) or data.dtype != np.dtype("<u2"):
            raise ValueError("Continuous feature identity mismatch")
        feature_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if record.get("feature_sha256") and record["feature_sha256"] != feature_hash:
            raise ValueError("Continuous feature checksum mismatch")
        detector.reset()
        scores = smooth(detector.predict(data), smoothing_steps)
        np.save(args.output/f"scores-{index:03d}.npy", scores)
        count = events(scores, threshold)
        identity = {"speaker": record["speaker"]} if "speaker" in record else {"id": record["id"]}
        rows.append({**identity, "seconds": record["seconds"],
            "speech_seconds": record["speech_seconds"], "events": count,
            "feature_sha256": feature_hash,
            "pcm_sha256": record["pcm_sha256"]})
        print(json.dumps({"completed": index+1, "total": len(corpus["records"]), "events": count,
                          "seconds": round(time.monotonic()-started, 1)}), flush=True)

    def summary(count, seconds):
        hours = seconds/3600
        return {"events": count, "hours_including_separators": hours,
                "false_per_hour": count/hours,
                "false_per_hour_upper_95": float(.5*chi2.ppf(.95, 2*(count+1))/hours)}

    total_seconds = sum(r["seconds"] for r in rows)
    result = {"model_sha256": model_hash, "threshold": threshold,
        "smoothing_steps": smoothing_steps,
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "calibration_scope": calibration.get("selection_scope"),
        "validation_operating_point": calibration.get("viable_operating_point"),
        "corpus_manifest_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "hey_nyx": summary(sum(r["events"] for r in rows), total_seconds),
        "speech_hours": sum(r["speech_seconds"] for r in rows)/3600,
        "standalone_host_seconds": round(time.monotonic()-started, 1),
        "records": rows, "limits": corpus["limits"]+[
            "Host BUILTIN_REF INT8 inference, not P4 timing or full-app resource acceptance.",
            "Poisson upper bounds describe this measured distribution, not every household environment."]}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        manifest = args.corpus/"baseline-manifest.jsonl"
        if (not baseline.get("restored") or len(baseline["clips"]) != len(rows)
                or baseline["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()):
            raise ValueError("Hi ESP baseline incomplete or not restored, or manifest mismatch")
        for a, b in zip(baseline["clips"], rows):
            if a["sha256"] != b["pcm_sha256"] or a["samples"] != round(b["seconds"]*16000) or not a["result"]["complete"]:
                raise ValueError("Hi ESP PCM identity mismatch")
        result["hi_esp"] = summary(sum(r["result"]["events"] for r in baseline["clips"]), total_seconds)
        result["hi_esp_probe_sha256"] = baseline["probe_sha256"]
    (args.output/"report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("hey_nyx", "hi_esp") if k in result}), flush=True)


if __name__ == "__main__":
    main()
