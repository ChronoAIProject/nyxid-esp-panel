"""Freeze a candidate choice using float validation reports only, before testing."""
import argparse
import hashlib
import json
from pathlib import Path


def rank(metrics):
    if metrics["ambient_faph"] <= .5 and metrics["max_source_false_rate"] <= .01:
        return (1, metrics["macro_recall"], -metrics["ambient_faph"], -metrics["max_source_false_rate"])
    return (0, -metrics["ambient_faph"], -metrics["max_source_false_rate"], metrics["macro_recall"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True,
                        help="Model directory containing training.json and hey_nyx.tflite")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = []
    for directory in args.candidate:
        report = json.loads((directory / "training.json").read_text())
        digest = hashlib.sha256((directory / "hey_nyx.tflite").read_bytes()).hexdigest()
        if digest != report["model_sha256"]:
            raise ValueError(f"Model hash mismatch: {directory.name}")
        checkpoint = max(report["history"], key=lambda item: rank(item["selected"]))
        candidates.append({"name": directory.name, "model_sha256": digest,
                           "parameters": report["parameters"], "validation": checkpoint,
                           "rank": rank(checkpoint["selected"])})
    if len({item["name"] for item in candidates}) != len(candidates):
        raise ValueError("Candidate directory names must be unique")
    selected = max(candidates, key=lambda item: (*item["rank"], -item["parameters"]))
    result = {"selection": "Frozen using float validation only; no final test audio read for candidate choice. Prefer viable mean-source recall; otherwise ambient FAR, then maximum source confusing-clip rate, then mean-source recall.",
              "selected": selected["name"], "candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(selected["name"])


if __name__ == "__main__":
    main()
