"""Independently inspect frozen GPU diagnostic sources and compare constructed event traces."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np

from float32_event_policy_mac_reference_v1 import event_steps as reference

PIN = "e3c4705f1e10b5cca6028d8fbaf7535c43c9ca31"
EVENT_SHA = "290a8725360f4b46bc05a08efb95fc4fd4bec6764dea89243c3b229e5b4d8b5d"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()

    def public(name):
        return subprocess.check_output(["git", "show", PIN + ":research/wakeword_p4/" + name], cwd=a.repository_root)

    lock_bytes = public("results/4090-fp32-reused-development-execution-v1.json")
    lock = json.loads(lock_bytes)
    scope_bytes = public("results/4090-fp32-reused-development-prospective-v1.json")
    scope = json.loads(scope_bytes)
    source_bytes = public("experiments/fp32_event_policy_v1.py")
    assert sha(source_bytes) == EVENT_SHA
    event_module = types.ModuleType("pinned_gpu_fp32_policy")
    exec(compile(source_bytes, "pinned-gpu-fp32-policy", "exec"), event_module.__dict__)
    sources = {}
    for name in ["fp32_event_policy_v1.py", "collect_fp32_reused_development_v1.py",
                 "score_fp32_reused_development_v1.py", "hey_development_metrics_v1.py"]:
        data = public("experiments/" + name)
        digest = sha(data)
        locked = [(key, value) for key, value in lock["files"].items() if key.endswith("/" + name)]
        assert locked and all(value == digest for key, value in locked)
        sources[name] = digest
    scorer = ast.parse(public("experiments/score_fp32_reused_development_v1.py"))
    selector_calls = [node for node in ast.walk(scorer) if isinstance(node, ast.Call) and
                      ((isinstance(node.func, ast.Name) and node.func.id == "choose") or
                       (isinstance(node.func, ast.Attribute) and node.func.attr == "choose"))]
    assert not selector_calls
    assert lock["models"] == scope["models"] and lock["grid"] == scope["grid"]
    prior = json.loads((a.repository_root / "research/wakeword_p4/results/mac-fp32-internal-trained-intake-v1.json").read_text())
    assert {m["target"] + "-" + m["condition"]: m["sha256"] for m in lock["models"]} == {
        name: row["model_sha256"] for name, row in prior["contracts"].items()}
    assert {k: row["records"] for k, row in lock["materials"].items()} == dict(synthetic=360, background=12)
    assert all(row["role"] == "reused-development" for row in lock["materials"].values())
    assert lock["threshold_selected"] is lock["acceptance_scored"] is False and lock["retries"] == 0
    assert len(lock["grid"]) == 138
    rng = np.random.default_rng(20261114)
    traces = [np.zeros(512, np.float32), np.ones(512, np.float32),
              np.full(512, np.float32(.501), np.float32),
              np.array(([1] * 5 + [0] * 67 + [1] * 5) * 5, np.float32),
              np.array([1] * 5 + [0] * 10 + [1] * 100, np.float32),
              np.array([0, np.nextafter(np.float32(0), np.float32(1)), 1, .5, 0] * 50, np.float32)]
    traces += [rng.random(512, dtype=np.float32) for _ in range(6)]
    extra = [float(np.float32(.501)), 1.0]
    checks = 0
    for raw in traces:
        before = raw.tobytes()
        for threshold in [*lock["grid"], *extra]:
            assert event_module.event_steps(raw, threshold) == reference(raw, threshold)
            checks += 1
        assert raw.tobytes() == before
    scipy_bound = "scipy" in lock["versions"] or any("scipy" in name.lower() for name in lock["files"])
    result = dict(schema="mac-gpu-raw-fp32-reused-protocol-source-review-v1", gpu_commit=PIN,
                  execution_sha256=sha(lock_bytes), prospective_sha256=sha(scope_bytes), source_sha256=sources,
                  constructed_valid_float32_event_trace_comparisons=checks, registered_points=138,
                  extra_constructed_thresholds=extra, arithmetic_equivalent_on_constructed_valid_domain=True,
                  original_model_identities_unchanged=True, source_workpoint_selector_calls=0,
                  material_roles="reused-development", synthetic_clips=360, background_streams=12,
                  scipy_identity_bound_in_original_execution=scipy_bound,
                  postcollection_checks_required=[
                      "Authenticate original result model report SHA and JSONL receipts against published original bytes.",
                      "Compare every complete report record/phase/order against original authenticated manifests, not len372 alone.",
                      "Check both probability dtypes/shapes/finiteness/repeats, raw original clocks and trace identities independently.",
                      "Record scoring dependency versions, including SciPy, in an additive receipt without changing the original collection lock.",
                      "Independently recompute events, group denominators, all original exposure/FAR and signed boundary summaries before interpreting quality."
                  ],
                  original_collection_restart_required=False, collection_sources_changed=False,
                  model_invocations=0, media_reads=0, real_probability_reads=0, thresholds_selected=False,
                  reviewer_sha256=sha(Path(__file__).read_bytes()),
                  reference_sha256=sha(Path(__file__).with_name("float32_event_policy_mac_reference_v1.py").read_bytes()),
                  limits="Static frozen-source and constructed-policy comparisons, not observed GPU liveness, raw collection completion, saved actual metric verification or recognition/P4 qualification. GPU policy is value-generic; actual scorer separately enforces original float32 vectors. Scoring reader postcollection checks do not imply corrupted data or a failed original collection.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
