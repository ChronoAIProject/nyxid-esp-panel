"""Authenticate completed training features and freeze both fitting plans."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STAGE = HERE / "plans/us-embedding-continuation-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh report path required")
    stage_bytes = STAGE.read_bytes()
    stage = json.loads(stage_bytes)
    models = args.models.resolve()
    require({n: importlib.metadata.version(n) for n in stage["versions"]} == stage["versions"],
            "Frozen runtime differs")
    require(sha((HERE.parent / "wakeword_p4/embedding/train_head.py").read_bytes())
            == stage["trainer_sha256"], "Current matched trainer differs")
    control_bytes = (models / stage["base_control"]).read_bytes()
    require(sha(control_bytes) == stage["base_control_sha256"], "Original control differs")
    control = json.loads(control_bytes)
    require(sha((models / stage["initial_weights"]).read_bytes()) == stage["initial_weights_sha256"],
            "Starting weights differ")
    for name, entry in control["feature_inputs"].items():
        require(sha((models / name / "index.jsonl").read_bytes()) == entry["index_sha256"],
                "Original feature index differs")
    base = models / stage["feature_directory"]
    index_bytes = (base / "index.jsonl").read_bytes()
    complete_bytes = (base / "complete.json").read_bytes()
    rows = [json.loads(line) for line in index_bytes.splitlines()]
    jobs = {r["id"]: r for r in json.loads((base / "jobs-private.json").read_bytes())}
    diagnostic_report = json.loads((args.diagnostic / "report.json").read_bytes())
    old_row_bytes = (args.diagnostic / "rows.jsonl").read_bytes()
    require(sha(old_row_bytes) == diagnostic_report["rows_sha256"], "Diagnostic rows differ")
    old_rows = [json.loads(line) for line in old_row_bytes.splitlines()]
    old = {r["source_sha256"]: r for r in old_rows}
    require(len(old) == 108 and len(rows) == len(jobs) == 216, "Source or feature grid incomplete")
    require(json.loads(complete_bytes)["jobs"] == 216, "Feature completion differs")
    variants, quiet = Counter(), 0
    for row in rows:
        job = jobs[row["id"]]
        require(row["split"] == job["split"] == "training" and row["engine"] == stage["source_engine"]
                and row["source_sha256"] == job["source_sha256"] and row["label"] == job["label"],
                "Training feature identity differs")
        prior = old[row["source_sha256"]]
        require(row["label"] == int(prior["target"] == "hey_nyx"), "Training label differs")
        expected_end = (38400 + prior["source_samples"]) / 16000 if row["label"] else None
        require(row["keyword_end"] == expected_end,
                "Complete-source endpoint differs")
        path = Path(row["path"])
        require(path.resolve().is_relative_to(base.resolve()), "Feature path outside expected data root")
        require(sha(path.read_bytes()) == row["features_sha256"], "Saved training feature differs")
        value = np.load(path, allow_pickle=False)
        require(value.dtype == np.int8 and value.shape == (row["frames"], 96), "Feature shape differs")
        variants[row["variant"]] += 1
        if row["variant"] == 0:
            require(sha((args.diagnostic / prior["arrays"]).read_bytes()) == prior["arrays_sha256"],
                    "Original diagnostic arrays differ")
            with np.load(args.diagnostic / prior["arrays"], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(value, arrays["embeddings"])
            quiet += 1
    require(variants == {0: 108, 1: 108} and quiet == 108, "Variant grid differs")
    extra = dict(directory=stage["feature_directory"], index_sha256=sha(index_bytes),
                 complete_sha256=sha(complete_bytes))
    outputs = {}
    for arm in ("control", "treatment"):
        plan = copy.deepcopy(control["plan"])
        plan.update(scope=stage["scope"], arm=arm, continuation_plan_sha256=sha(stage_bytes),
                    comparison=stage["only_arm_difference"], mining_policy="No mining in either arm")
        if arm == "treatment":
            plan["extra_training_features"].append(extra)
        path = HERE / "plans" / ("us-embedding-dense-" + arm + "-v1.json")
        require(not path.exists(), "Final fitting plan already exists")
        path.write_text(json.dumps(plan, indent=2) + "\n")
        outputs[path.name] = sha(path.read_bytes())
    result = dict(scope="Completed training-feature identity and quiet-render equivalence; not wake acceptance",
                  stage_plan_sha256=sha(stage_bytes), script_sha256=sha(Path(__file__).read_bytes()),
                  original_control_sha256=sha(control_bytes), new_index_sha256=sha(index_bytes),
                  new_complete_sha256=sha(complete_bytes), verified_training_features=216,
                  quiet_embeddings_exact_to_original_diagnostic=quiet, variants=dict(variants),
                  fitting_plans_sha256=outputs, no_model_fitting_performed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
