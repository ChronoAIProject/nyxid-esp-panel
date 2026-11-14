"""Freeze the existing synthetic P4 image and verify only capture validation."""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--original-run", type=Path, required=True)
    p.add_argument("--constructed-report", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    a = p.parse_args()
    repo, original = a.repository_root, a.original_run
    prefix = repo / "research/wakeword_p4"
    runner = prefix / "run_probe.py"
    parsed = ast.parse(runner.read_text())
    helper, = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "validate_fp32_result"]
    namespace = dict(math=math)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<capture-validator-only>", "exec"), namespace)
    validate = namespace["validate_fp32_result"]
    valid = dict(fp32_models=[dict(condition=i, repeat=j, passed=True, max_abs_error=1e-4,
        arena_used=256 * 1024, variable_arena_allocated=64 * 1024,
        p50_us=1, p95_us=30000, max_us=30000) for i in range(4) for j in range(2)],
        FP32_SYNTHETIC_DONE=dict(passed=True, stack_free=1024))
    validate(valid)
    bad = {}
    def changed(name, key, value):
        item = copy.deepcopy(valid); item["fp32_models"][0][key] = value; bad[name] = item
    for name, key, value in [
        ("case_failure", "passed", False), ("nan_error", "max_abs_error", float("nan")),
        ("infinite_time", "max_us", float("inf")), ("negative_error", "max_abs_error", -1),
        ("error_above_ceiling", "max_abs_error", math.nextafter(1e-4, math.inf)),
        ("zero_arena", "arena_used", 0), ("oversized_arena", "arena_used", 256 * 1024 + 1),
        ("wrong_variable_capacity", "variable_arena_allocated", 65535), ("boolean_arena", "arena_used", True),
        ("boolean_condition", "condition", False), ("zero_duration", "p50_us", 0),
        ("invalid_quantile_order", "p50_us", 30001), ("missed_hop_deadline", "max_us", 30001)]:
        changed(name, key, value)
    item = copy.deepcopy(valid); item["fp32_models"].pop(); bad["missing_case"] = item
    item = copy.deepcopy(valid); item["fp32_models"][0]["condition"] = 3; item["fp32_models"][0]["repeat"] = 1; bad["duplicate_case"] = item
    item = copy.deepcopy(valid); item["FP32_SYNTHETIC_DONE"]["passed"] = False; bad["done_failure"] = item
    item = copy.deepcopy(valid); del item["FP32_SYNTHETIC_DONE"]; bad["missing_done"] = item
    item = copy.deepcopy(valid); del item["fp32_models"][0]["max_us"]; bad["missing_measurement"] = item
    for name, item in bad.items():
        try:
            validate(item)
        except (ValueError, KeyError):
            continue
        raise AssertionError("accepted malformed capture: " + name)
    checks = dict(schema="mac-fp32-p4-resource-constructed-v1", script_sha256=sha(Path(__file__)),
                  runner_sha256=sha(runner), all_passed=True, exact_numeric_and_deadline_boundaries_accepted=True,
                  rejected_cases=list(bad), rejected_count=len(bad),
                  limits="Constructed capture-validator cases only. No serial/HTTP/flashing process, firmware execution, candidate or recognition evidence.")
    with a.constructed_report.open("x") as f: f.write(json.dumps(checks, indent=2) + "\n")
    base_plan_path = prefix / "results/mac-fp32-runtime-feasibility-plan-v1.json"
    base_plan = json.loads(base_plan_path.read_text())
    link_path = prefix / "results/mac-fp32-runtime-feasibility-p4-link-v1.json"
    linked = json.loads(link_path.read_text())
    binary = original / "build/nyx_fp32_feasibility.bin"
    assert sha(binary) == linked["image_sha256"] and binary.stat().st_size == linked["image_bytes"]
    probe = prefix / "experiments/fp32_runtime_feasibility_p4_v1.cpp"
    assert sha(probe) == sha(original / "project/main/probe.cpp") == base_plan["identity"]["p4_probe_sha256"]
    plan = dict(schema="mac-fp32-p4-resource-plan-v1", base_plan_sha256=sha(base_plan_path),
                original_link_report_sha256=sha(link_path), probe_bin_sha256=sha(binary), probe_bin_bytes=binary.stat().st_size,
                probe_cpp_sha256=sha(probe), runner_sha256=sha(runner),
                constructed_report_sha256=sha(a.constructed_report), planner_sha256=sha(Path(__file__)),
                input="The original four synthetic streams, 256steps each, two fresh/reset interpreters per stream; no audio/sensor/network in probe",
                preflight="Use only explicitly identified development USB board. Fresh LAN app/status identity, idle voice/OTA and connected service, current active slot/offset/size verified from partition information, current matching original-app backup. Expected MAC and status URL are required execution inputs, never defaults. No production board or full-project flashing.",
                execution="Existing run_probe.py --fp32-witness; app-only write at verified active slot. Always restore in finally, including numeric failure, allocation failure, timeout or cancellation; independently recheck original app/slot/service and wake state. Keep raw UART and original receipts private, publish only filtered numeric report.",
                fixed_requirements=dict(cases=8, steps_per_case=256, input_hop_us=30000,
                    max_abs_error_ceiling=1e-4, all_max_invoke_us_ceiling=30000,
                    main_arena_capacity=256 * 1024, variable_arena_capacity=64 * 1024,
                    all_finite=True, all_cases_and_terminal_passed=True),
                report="Actual P4 AllocateTensors/Invoke outcome, every cold/warm case p50/p95/max/error/main arena usage and variable allocation ceiling; raw log/hash and restoration receipt. Independent panel restoration check is required.",
                limitations="Standalone random-weight resource feasibility only. P4 first-call/init cost is retained; no startup exclusion. The 30ms ceiling checks ability to keep up with the original hop, not full-panel CPU headroom. Does not measure microphone/AEC/AGC/FIFO, recognition/FAR, real trained models, panel load or independent acceptance, select FP32 deployment, or alter old integer gates. No board has been accessed by this planning command.")
    with a.plan.open("x") as f: f.write(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(dict(plan_sha256=sha(a.plan), constructed_rejections=len(bad), board_accessed=False)))


if __name__ == "__main__":
    main()
