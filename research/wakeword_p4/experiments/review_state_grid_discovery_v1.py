"""Exercise state-grid discovery against both actual published model graphs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from infer_quantized_state_grids_v1 import state_grid_layout


def review(ref):
    ref = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    path = "research/wakeword_p4/results/4090-hey-broad-equalized-export-result-v1.json"
    raw = subprocess.check_output(["git", "show", f"{ref}:{path}"])
    exported = json.loads(raw)
    expected = {
        "hey_willow": [[0, 42, 48], [50, 41, 51], [53, 43, 54], [56, 44, 57], [59, 45, 60], [62, 46, 63]],
        "hey_nyx": [[0, 43, 48], [50, 44, 51], [53, 45, 54], [56, 46, 57], [59, 41, 60], [62, 42, 63]],
    }

    def run(arm):
        tensors = {t["index"]: t for t in arm["tensors"]}
        def grid(index):
            t = tensors[index]
            return dict(scales=[t["scale"]], zeros=[t["zero"]], dtype=t["dtype"], shape=t["shape"])
        return state_grid_layout(arm["ops"], grid)

    cases = []
    rejected = []
    for arm in exported["arms"]:
        layout = run(arm)
        assert layout["state_grid_pairs"] == expected[arm["target"]]
        assert layout["reshape_grid_pairs"] == [[63, 64]]
        # READ op order is not state identity.
        reordered = copy.deepcopy(arm)
        positions = [i for i, op in enumerate(reordered["ops"]) if op["op"] == "READ_VARIABLE"]
        operations = [reordered["ops"][i] for i in reversed(positions)]
        for at, op in zip(positions, operations):
            reordered["ops"][at] = op
        assert run(reordered) == layout
        # Tensor IDs are local to an export, including constants and handles.
        renamed = copy.deepcopy(arm)
        mapping = {t["index"]: 1000 + 7 * (74 - t["index"]) for t in renamed["tensors"]}
        for t in renamed["tensors"]:
            t["index"] = mapping[t["index"]]
        for op in renamed["ops"]:
            for key in ("inputs", "outputs"):
                op[key] = [mapping[i] for i in op[key]]
        assert run(renamed)["state_grid_pairs"] == [[mapping[i] for i in p] for p in expected[arm["target"]]]
        for fault in ("read_grid", "assignment_handle", "concat_time_order"):
            broken = copy.deepcopy(arm)
            if fault == "read_grid":
                read = layout["state_grid_pairs"][0][1]
                next(t for t in broken["tensors"] if t["index"] == read)["scale"] *= 2
            elif fault == "assignment_handle":
                writes = [o for o in broken["ops"] if o["op"] == "ASSIGN_VARIABLE"]
                writes[0]["inputs"][0], writes[1]["inputs"][0] = writes[1]["inputs"][0], writes[0]["inputs"][0]
            else:
                next(o for o in broken["ops"] if o["op"] == "CONCATENATION")["inputs"].reverse()
            try:
                run(broken)
            except (AssertionError, ValueError, KeyError):
                rejected.append(dict(target=arm["target"], fault=fault))
            else:
                raise AssertionError((arm["target"], "accepted mutation", fault))
        cases.append(dict(target=arm["target"], model_sha256=arm["sha256"], **layout,
                          reordered_reads_same=True, renumbered_tensors_same_relationships=True))
    return dict(schema="mac-state-grid-discovery-review-v1", gpu_ref=ref,
                export_result_sha256=hashlib.sha256(raw).hexdigest(),
                helper_sha256=hashlib.sha256(Path(__file__).with_name("infer_quantized_state_grids_v1.py").read_bytes()).hexdigest(),
                cases=cases, rejected_mutations=rejected,
                limits="Actual published topology/quantization metadata only, plus explicitly constructed mutations. No NPZ read, model invocation, QAT run, optimizer, new data or admission. This fixes export-local state identity discovery, not the original numerical error.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ref", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.gpu_ref)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(actual_models=len(result["cases"]), read_reorder_cases=len(result["cases"]),
                          tensor_renumber_cases=len(result["cases"]), rejected_mutations=len(result["rejected_mutations"]))))
