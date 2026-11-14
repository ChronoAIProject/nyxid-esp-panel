"""Independent struct/binary64 audit of every original native fullstream output."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def floats(array):
    return [p[0] for p in struct.iter_unpack("<f", array.astype("<f4").tobytes())]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    frozen = json.loads(a.plan.read_text())
    native, packet = a.native_root, a.packet_root
    original = json.loads((native / "report.json").read_text())
    assert original["plan_sha256"] == sha(a.plan.read_bytes())
    assert original["executable_sha256"] == frozen["native_build"]["executable_sha256"] == sha((native / "fp32-internal-fullstream").read_bytes())
    assert len(original["cases"]) == len(frozen["cases"]) == 540
    assert [json.loads(line) for line in (native / "cases.jsonl").read_text().splitlines()] == original["cases"]
    for name, digest in frozen["packet_files"].items():
        assert sha((packet / name).read_bytes()) == digest
    for path in (native / "models").glob("*.tflite"):
        assert sha(path.read_bytes()) == frozen["packet_files"]["models/" + path.name]
    counts, model_summaries, case_checks = 0, {}, []
    for spec, recorded in zip(frozen["cases"], original["cases"], strict=True):
        name, index, steps = spec["model"], spec["index"], spec["steps"]
        assert recorded["model"] == name and recorded["index"] == index and recorded["request_order"] == spec["request_order"]
        assert recorded["steps_per_repeat"] == steps and recorded["actual_native_invocations"] == steps * 2
        folder = native / (name + f"-{index:03}")
        assert json.loads((folder / "report.json").read_text()) == recorded
        data = (folder / "outputs.bin").read_bytes()
        assert len(data) == steps * 2 * 4 and sha(data) == recorded["outputs_sha256"]
        assert data[:steps * 4] == data[steps * 4:]
        values = [v[0] for v in struct.iter_unpack("<f", data)]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert sha((folder / "input.bin").read_bytes()) == spec["input_sha256"] == recorded["input_sha256"]
        receipt_bytes = (folder / "execution.json").read_bytes()
        assert sha(receipt_bytes) == recorded["execution_receipt_sha256"]
        receipt = json.loads(receipt_bytes)
        assert receipt["returncode"] == 0
        assert receipt["args"][0].endswith("/fp32-internal-fullstream")
        assert receipt["args"][1].endswith("/models/" + name + ".tflite")
        assert receipt["args"][2].endswith("/" + name + f"-{index:03}/input.bin")
        assert receipt["args"][3].endswith("/" + name + f"-{index:03}/outputs.bin")
        lines = receipt["stdout"].splitlines()
        assert lines.count("HOST_TFLM_STREAM_DONE") == 1
        streams = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("HOST_TFLM_STREAM ")]
        assert streams == recorded["streams"]
        assert [(r["repeat"], r["steps"]) for r in streams] == [(0, steps), (1, steps)]
        with np.load(packet / spec["member"], allow_pickle=False) as z:
            assert (folder / "input.bin").read_bytes() == z["input"].astype("<f4").tobytes()
            tf = floats(z["tf"]) + floats(z["tf_repeat"])
            lite = floats(z["litert"]) + floats(z["litert_repeat"])
            phase, nominal = floats(z["offline_phase"]), floats(z["offline_nominal"])
        assert len(tf) == len(lite) == len(values) == steps * 2
        assert len(phase) == len(nominal) == steps - 74
        gaps = dict(tflm_vs_tf_max_abs=max(abs(v - r) for v, r in zip(values, tf, strict=True)),
                    tflm_vs_litert_max_abs=max(abs(v - r) for v, r in zip(values, lite, strict=True)),
                    tflm_vs_phase_max_abs=max(abs(values[repeat * steps + 74 + i] - r) for repeat in range(2) for i, r in enumerate(phase)),
                    nominal225_difference_descriptive=max(abs(values[repeat * steps + 74 + i] - r) for repeat in range(2) for i, r in enumerate(nominal)))
        assert all(gap == recorded[k] for k, gap in gaps.items())
        assert max(gaps[k] for k in gaps if k != "nominal225_difference_descriptive") <= frozen["max_abs"] == 1e-4
        assert recorded["passed"] and recorded["fresh_repeats_byte_exact"] and recorded["finite_probability"]
        summary = model_summaries.setdefault(name, dict(cases=0, native_invocations=0, tf_max_abs=0.0,
                                                        litert_max_abs=0.0, phase_max_abs=0.0,
                                                        fresh_repeats_byte_exact=True, main_arena_bytes=set(), variable_allocator_bytes=set()))
        summary["cases"] += 1
        summary["native_invocations"] += steps * 2
        for label in ["tf", "litert", "phase"]:
            summary[label + "_max_abs"] = max(summary[label + "_max_abs"], gaps["tflm_vs_" + label + "_max_abs"])
        summary["main_arena_bytes"].update(r["arena_used"] for r in streams)
        summary["variable_allocator_bytes"].update(r["variable_used"] for r in streams)
        counts += steps * 2
        case_checks.append(dict(model=name, index=index, audited_probabilities=steps * 2,
                                outputs_sha256=sha(data), **gaps))
    assert counts == original["actual_native_invocations"] == frozen["expected_native_invocations"] == 192336
    assert original["all_passed"] and len(model_summaries) == 4
    for summary in model_summaries.values():
        assert summary["cases"] == 135
        for key in ["main_arena_bytes", "variable_allocator_bytes"]:
            summary[key] = sorted(summary[key])
    report = dict(schema="mac-fp32-internal-fullstream-host-saved-byte-audit-v1", all_passed=True,
                  plan_sha256=sha(a.plan.read_bytes()), original_result_sha256=sha((native / "report.json").read_bytes()),
                  cases_checked=540, original_probabilities_audited=counts, summaries=model_summaries,
                  case_checks=case_checks, model_invocations=0, auditor_sha256=sha(Path(__file__).read_bytes()),
                  limits="Independent scalar struct/binary64 reductions of all original native bytes, both repeats/all startup, against unchanged GPU TF/LiteRT/phase arrays, and exact nominal descriptive reductions. Complete packet/executable/input/output/execution identities verified. No re-inference or PCM/frontend/H5 reproduction. Numerical fullstream consistency is not recognition/FAR/P4/acceptance qualification.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["all_passed", "cases_checked", "original_probabilities_audited", "summaries", "model_invocations"]}))


if __name__ == "__main__":
    main()
