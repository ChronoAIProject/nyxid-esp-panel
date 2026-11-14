"""Independently check raw captured float bytes and original fixed references."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--prior-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    run = a.run_root
    plan = json.loads(a.plan.read_text())
    report = json.loads((run / "report.json").read_text())
    assert report["plan_sha256"] == sha(a.plan)
    assert sha(a.prior_root / "model.tflite") == report["original_model_sha256"] == plan["original_model_sha256"]
    assert sha(a.prior_root / "synthetic-arrays.npz") == report["original_arrays_sha256"] == plan["original_arrays_sha256"]
    for filename, key in [("compile.json", "compile_receipt_sha256"), ("execution.json", "execution_receipt_sha256"),
                          ("saved-arrays.npz", "saved_arrays_sha256"), ("inputs.bin", "inputs_sha256"),
                          ("outputs.bin", "outputs_sha256"), ("fp32-real-tflm", "executable_sha256")]:
        assert sha(run / filename) == report[key]
    compile = json.loads((run / "compile.json").read_text())
    executed = json.loads((run / "execution.json").read_text())
    assert compile["returncode"] == executed["returncode"] == 0
    sources = [Path(s) for s in compile["args"] if s.endswith(".cc")]
    assert len(sources) == len(plan["source_sha256"]) == 56
    assert {str(s).split("/espressif__esp-tflite-micro/", 1)[1]: sha(s) for s in sources} == plan["source_sha256"]
    assert all(flag in compile["args"] for flag in plan["flags"])
    raw = (run / "outputs.bin").read_bytes()
    values = struct.unpack("<2048f", raw)
    max_tf = max_lite = 0.
    with np.load(a.prior_root / "synthetic-arrays.npz", allow_pickle=False) as original, np.load(run / "saved-arrays.npz", allow_pickle=False) as saved:
        inputs = b""
        for i, name in enumerate(plan["streams"]):
            inputs += original[name + "_input"].astype("<f4").tobytes()
            assert raw[(i * 2) * 1024:(i * 2 + 1) * 1024] == raw[(i * 2 + 1) * 1024:(i * 2 + 2) * 1024]
            assert saved["tf"][i].tobytes() == original[name + "_tf"].tobytes()
            assert saved["lite"][i].tobytes() == original[name + "_lite"].tobytes()
            for j in range(2):
                expected = values[(i * 2 + j) * 256:(i * 2 + j + 1) * 256]
                assert saved["tflm"][i, j].astype("<f4").tobytes() == struct.pack("<256f", *expected)
                max_tf = max(max_tf, *(abs(v - float(r)) for v, r in zip(expected, original[name + "_tf"], strict=True)))
                max_lite = max(max_lite, *(abs(v - float(r)) for v, r in zip(expected, original[name + "_lite"], strict=True)))
        assert inputs == (run / "inputs.bin").read_bytes()
    assert max_tf == report["tflm_vs_tf_max_abs"] and max_lite == report["tflm_vs_lite_max_abs"]
    assert max(max_tf, max_lite) <= plan["float_max_abs_ceiling"]
    case_lines = [json.loads(s.split(" ", 1)[1]) for s in executed["stdout"].splitlines() if s.startswith("HOST_TFLM_CASE ")]
    assert case_lines == report["cases"] and len(case_lines) == 8
    result = dict(schema="mac-fp32-real-tflm-host-saved-byte-audit-v1", script_sha256=sha(Path(__file__)),
                  original_result_sha256=sha(run / "report.json"), actual_translation_unit_hashes_exact=56,
                  original_input_bytes_exact=True, captured_output_floats_exact=2048,
                  fresh_repeat_bytes_exact=True, original_tf_and_lite_references_exact=True,
                  tflm_vs_tf_max_abs=max_tf, tflm_vs_lite_max_abs=max_lite,
                  actual_saved_allocator_records=case_lines, all_checks_pass=True,
                  limits="Independent byte/struct-unpack check of original host execution. No second inference, new data, candidate qualification, P4 allocation or latency evidence.")
    with a.report.open("x") as f: f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
