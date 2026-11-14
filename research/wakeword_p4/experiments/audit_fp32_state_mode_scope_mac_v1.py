"""Inspect the bound state-mode factory and host ABI; no ML runtime imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repository_root
    export = json.loads((repo / "research/wakeword_p4/results/mac-fp32-runtime-feasibility-export-v1.json").read_text())
    raw = (args.upstream / "microwakeword/utils.py").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == export["identity"]["upstream_python_sha256"]["microwakeword/utils.py"]
    function = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "convert_to_inference_model")
    node = next(n for n in function.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name) and n.test.left.id == "mode")
    conditions = []
    while isinstance(node, ast.If):
        conditions.append(ast.unparse(node.test))
        node = node.orelse[0]
    assert conditions == ["mode == modes.Modes.STREAM_INTERNAL_STATE_INFERENCE",
                          "mode == modes.Modes.NON_STREAM_INFERENCE"]
    assert isinstance(node, ast.Raise) and "non supported mode" in ast.unparse(node)
    cpp = (repo / "research/wakeword_p4/experiments/fp32_real_tflm_host_v1.cpp").read_bytes()
    cpp_digest = hashlib.sha256(cpp).hexdigest()
    plan = json.loads((repo / "research/wakeword_p4/results/mac-fp32-real-tflm-host-plan-v1.json").read_text())
    assert cpp_digest == plan["probe_sha256"]
    assert b"auto *in = interpreter.input(0); auto *out = interpreter.output(0);" in cpp
    assert b"in->bytes != 480" in cpp and b"out->bytes != 4" in cpp
    scope_ref = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "d652049b"], text=True).strip()
    result = dict(schema="mac-fp32-state-mode-scope-source-audit-v1", upstream_utils_sha256=digest,
                  bound_upstream_source_exact=True, factory_function="convert_to_inference_model",
                  supported_factory_branches=conditions, external_factory_branch_present=False,
                  unmatched_mode_raises=ast.unparse(node), current_real_host_probe_sha256=cpp_digest,
                  bound_host_source_exact=True, current_host_inputs="one float32[1,3,40] audio tensor",
                  current_host_output="one float32 probability tensor", proposed_external_scope_ref=scope_ref,
                  source_only=True, model_invocations=0, media_reads=0,
                  limits="Static inspection of exact pinned sources. External mode enum/layers exist but factory conversion rejects unmatched mode; no external adapter is implemented or tested by the current sources. Proposed scope is not an executed failure. Existing internal-state export/runtime evidence does not prove external-state ABI. Prefer original internal-state route or separately implement/test external adapter before any such claim.")
    with args.report.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
