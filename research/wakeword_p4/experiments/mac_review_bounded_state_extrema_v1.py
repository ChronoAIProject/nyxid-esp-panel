"""Review published failure-trace extrema, without model/data inference.

This checks necessary state-transport invariants, not complete tensor equality.
The saved NPZ arrays remain on GPU and are not independently audited here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = "research/wakeword_p4/results/"


def review(ref):
    ref = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    artifacts = {}

    def read(name):
        path = ROOT + name
        raw = subprocess.check_output(["git", "show", f"{ref}:{path}"])
        artifacts[path] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    exported = read("4090-hey-broad-bounded-export-result-v1.json")
    plan = read("4090-hey-broad-bounded-tensors-plan-v1.json")
    traced = read("4090-hey-broad-bounded-tensors-result-v1.json")
    assert traced["plan_sha256"] == artifacts[ROOT + "4090-hey-broad-bounded-tensors-plan-v1.json"]
    assert plan["files"][ROOT + "4090-hey-broad-bounded-export-result-v1.json"] == artifacts[ROOT + "4090-hey-broad-bounded-export-result-v1.json"]
    assert {c["target"] for c in traced["cases"]} == {"hey_willow", "hey_nyx"}
    cases = []
    for case in traced["cases"]:
        target = case["target"]
        arm, = [a for a in exported["arms"] if a["target"] == target]
        selected, = [c for c in plan["cases"] if c["target"] == target]
        assert case["request_order"] == selected["request_order"]
        assert plan["exports"][target] == arm["path"]
        assert plan["files"][arm["path"]] == arm["sha256"]
        tensors = {t["index"]: t for t in case["tensors"]}
        assert len(tensors) == len(case["tensors"])
        steps, = {len(t["per_step_min"]) for t in tensors.values()}
        assert steps > 1
        for t in tensors.values():
            assert len(t["per_step_max"]) == steps
            assert all(a <= b for a, b in zip(t["per_step_min"], t["per_step_max"]))
            assert min(t["per_step_min"]) == t["minimum"]
            assert max(t["per_step_max"]) == t["maximum"]
        reads = {o["inputs"][0]: o["outputs"][0] for o in arm["ops"] if o["op"] == "READ_VARIABLE"}
        writes = {o["inputs"][0]: o["inputs"][1] for o in arm["ops"] if o["op"] == "ASSIGN_VARIABLE"}
        assert reads.keys() == writes.keys() and len(reads) == 6
        states = []
        for handle, index in reads.items():
            src, dst = tensors[index], tensors[writes[handle]]
            for key in ("shape", "dtype", "scale", "zero"):
                assert src[key] == dst[key], (target, handle, key)
            for key in ("per_step_min", "per_step_max"):
                assert src[key][1:] == dst[key][:-1], (target, handle, key)
            states.append({
                "handle": handle, "read": index, "assigned": writes[handle],
                "shape": src["shape"], "scale": src["scale"], "zero": src["zero"],
                "first_visible_read_extrema": [src["per_step_min"][0], src["per_step_max"][0]],
                "matching_next_step_extrema_pairs": steps - 1,
            })
        concats = []
        reshapes = []
        for op in arm["ops"]:
            if op["op"] == "CONCATENATION":
                inputs = [tensors[i] for i in op["inputs"]]
                out = tensors[op["outputs"][0]]
                for t in inputs:
                    for key in ("dtype", "scale", "zero"):
                        assert out[key] == t[key], (target, op, key)
                for at in range(steps):
                    assert out["per_step_min"][at] == min(t["per_step_min"][at] for t in inputs)
                    assert out["per_step_max"][at] == max(t["per_step_max"][at] for t in inputs)
                concats.append({"output": out["index"], "matching_extrema_pairs": steps})
            elif op["op"] == "RESHAPE":
                src, dst = tensors[op["inputs"][0]], tensors[op["outputs"][0]]
                for key in ("dtype", "scale", "zero", "per_step_min", "per_step_max"):
                    assert src[key] == dst[key], (target, op, key)
                reshapes.append({"output": dst["index"], "matching_extrema_pairs": steps})
        assert len(concats) == 6 and len(reshapes) == 1
        cases.append({
            "target": target, "request_order": case["request_order"], "steps": steps,
            "model_sha256": arm["sha256"], "gpu_npz_sha256_not_downloaded": case["arrays_sha256"],
            "states": states, "concatenations": concats, "reshapes": reshapes,
        })
    return {
        "schema": "mac-bounded-state-extrema-review-v1", "gpu_ref": ref,
        "artifacts": artifacts, "cases": cases,
        "state_transition_extrema_pairs": sum(sum(s["matching_next_step_extrema_pairs"] for s in c["states"]) for c in cases),
        "concatenation_extrema_pairs": sum(sum(s["matching_extrema_pairs"] for s in c["concatenations"]) for c in cases),
        "reshape_extrema_pairs": sum(sum(s["matching_extrema_pairs"] for s in c["reshapes"]) for c in cases),
        "all_published_extrema_invariants_consistent": True,
        "limits": (
            "Only published per-step min/max metadata from two selected training failures was read. "
            "Passing extrema invariants cannot prove exact byte transport, channel/time ordering, correct "
            "float-state equivalence or absence of state-related error. READ snapshots were taken after "
            "invoke, so first-visible nonzero extrema are not a pre-invoke initialization audit. "
            "No saved NPZ, model invocation, export, development, acceptance, playback, recording or board write. "
            "Original numerical rejection remains; this is not model admission."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.gpu_ref)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "artifacts", "limits")}))
