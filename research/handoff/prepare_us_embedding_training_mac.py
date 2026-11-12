"""Adapt the complete original US training cohort to the unchanged extractor."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PLAN = HERE / "plans/us-embedding-continuation-v1.json"
EXTRACTOR = HERE.parent / "wakeword_p4/embedding/prepare_head_features.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    require(not args.work.exists(), "Fresh private work directory required")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    require(sha(EXTRACTOR.read_bytes()) == plan["extractor_sha256"], "Extractor differs")
    source, models = args.source.resolve(), args.models.resolve()
    require({n: importlib.metadata.version(n) for n in plan["versions"]} == plan["versions"],
            "Frozen runtime differs")
    for name, key in [("native-fft-window.npy", "window_sha256"),
                      ("native-mel-filter.npy", "mel_filter_sha256")]:
        require(sha((models / name).read_bytes()) == plan[key], "Frontend artifact differs")
    manifest = (source / "manifest.jsonl").read_bytes()
    require(sha(manifest) == plan["source_manifest_sha256"], "Original manifest differs")
    rows = [json.loads(line) for line in manifest.splitlines()]
    require(len(rows) == plan["source_clips"], "Original cohort incomplete")
    jobs = []
    for order, row in enumerate(rows):
        require(row["order"] == order and row["split"] == "training" and row["quality_pass"]
                and row["role"] == "us-coverage-training-only", "Original training identity differs")
        relative = Path(row["path"])
        require(relative.parts == (relative.name,), "Invalid source path")
        path = source / relative
        require(sha(path.read_bytes()) == row["sha256"], "Original WAV differs")
        for variant in plan["feature_variants"]:
            identity = sha((row["path"] + ":" + str(variant) + ":us-embedding-continuation-v1").encode())
            jobs.append(dict(id=identity, kind="tts", split="training", label=int(row["target"] == "hey_nyx"),
                             source=str(path), source_sha256=row["sha256"], keyword_end=row["samples"],
                             variant=variant, engine=plan["source_engine"]))
    require(len(jobs) == len({r["id"] for r in jobs}) == plan["feature_jobs"], "Feature grid differs")
    args.work.mkdir(parents=True)
    (args.work / "plan.json").write_bytes(plan_bytes)
    (args.work / "source.py").write_bytes(Path(__file__).read_bytes())
    jobs_path = args.work / "jobs-private.json"
    jobs_path.write_text(json.dumps(jobs) + "\n")
    subprocess.run([sys.executable, str(EXTRACTOR), "--output", str(models / plan["feature_directory"]),
                    "--jobs", str(jobs_path), "--workers", "2", "--model",
                    str(models / "embedding-stream-int8-quiet.tflite"), "--expected-model-sha256",
                    plan["backbone_sha256"], "--window", str(models / "native-fft-window.npy"),
                    "--mel-filter", str(models / "native-mel-filter.npy")], check=True)


if __name__ == "__main__":
    main()
