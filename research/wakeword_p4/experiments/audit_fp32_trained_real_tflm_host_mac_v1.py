"""Independently reduce original native output bytes; never invoke a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(packet, native, frozen_path):
    frozen = json.loads(frozen_path.read_text())
    result = json.loads((native / "report.json").read_text())
    assert result["plan_sha256"] == sha(frozen_path.read_bytes())
    assert result["executable_sha256"] == sha((native / "fp32-trained-real-tflm").read_bytes())
    assert result["compile_receipt_sha256"] == sha((native / "compile.json").read_bytes())
    compiled = json.loads((native / "compile.json").read_text())
    assert compiled["returncode"] == 0
    assert compiled["args"][0] == frozen["compiler"]
    assert compiled["args"][1:1 + len(frozen["flags"])] == frozen["flags"]
    # Actual compile command includes every locked runtime translation unit once.
    for source in frozen["source_sha256"]:
        assert sum(a.endswith("/" + source) for a in compiled["args"]) == 1, source
    for name, expected in frozen["packet_files"].items():
        assert sha((packet / name).read_bytes()) == expected, name
    actual_rows = []
    for name in frozen["models"]:
        row = next(r for r in result["models"] if r["model"] == name)
        folder = native / name
        raw = (folder / "outputs.bin").read_bytes()
        assert len(raw) == 4 * 2 * 256 * 4 and sha(raw) == row["outputs_sha256"]
        assert sha((folder / "saved-arrays.npz").read_bytes()) == row["saved_arrays_sha256"]
        assert sha((folder / "execution.json").read_bytes()) == row["execution_receipt_sha256"]
        execution = json.loads((folder / "execution.json").read_text())
        assert execution["returncode"] == 0 and execution["args"][0].endswith("/fp32-trained-real-tflm")
        assert execution["args"][1].endswith("/" + name + "/model.tflite")
        assert execution["args"][2].endswith("/inputs.bin")
        assert execution["args"][3].endswith("/" + name + "/outputs.bin")
        lines = execution["stdout"].splitlines()
        assert lines.count("HOST_TFLM_DONE") == 1
        cases = [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("HOST_TFLM_CASE ")]
        assert cases == row["cases"]
        assert [(r["condition"], r["repeat"], r["steps"]) for r in cases] == [(c, r, 256) for c in range(4) for r in range(2)]
        native_values = [v[0] for v in struct.iter_unpack("<f", raw)]
        assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in native_values)
        # Each stream contains consecutive complete repeat0/repeat1 byte blocks.
        for stream in range(4):
            start = stream * 2048
            assert raw[start:start + 1024] == raw[start + 1024:start + 2048]
        gaps = {}
        with np.load(packet / name / "references.npz", allow_pickle=False) as original:
            for backend in ["tf", "litert"]:
                reference = original[backend]
                assert reference.shape == (4, 2, 256) and reference.dtype == np.float32
                reference_values = [v[0] for v in struct.iter_unpack("<f", reference.astype("<f4").tobytes())]
                errors = [abs(a - b) for a, b in zip(native_values, reference_values)]
                assert len(errors) == 2048
                maximum = max(errors)
                assert maximum <= frozen["float_max_abs_ceiling"]
                assert maximum == row["tflm_vs_" + backend + "_max_abs"]
                gaps[backend] = dict(max_abs=maximum, worst_flat_step=errors.index(maximum))
            with np.load(folder / "saved-arrays.npz", allow_pickle=False) as saved:
                assert set(saved.files) == {"tf", "litert", "tflm"}
                assert saved["tflm"].astype("<f4").tobytes() == raw
                assert all(saved[k].tobytes() == original[k].tobytes() for k in ["tf", "litert"])
        assert row["passed"] and row["fresh_repeats_byte_exact"] and row["finite"]
        assert row["model_sha256"] == frozen["packet_files"][name + "/model.tflite"]
        assert row["original_references_sha256"] == frozen["packet_files"][name + "/references.npz"]
        assert row["original_h5_sha256"] == frozen["h5_bindings"][name]
        actual_rows.append(dict(model=name, actual_saved_probabilities=2048, reference_errors=gaps,
                                repeats_byte_exact=True, arena_used=sorted({c["arena_used"] for c in cases}),
                                variable_allocator_used=sorted({c["variable_used"] for c in cases}),
                                outputs_sha256=sha(raw)))
    assert sum(r["actual_saved_probabilities"] for r in actual_rows) == frozen["expected_invocations"] == result["actual_registered_tflm_steps"] == 8192
    assert result["all_passed"] and len(result["models"]) == len(actual_rows) == 4
    return dict(schema="mac-fp32-trained-real-tflm-host-saved-byte-audit-v1", all_passed=True,
                plan_sha256=sha(frozen_path.read_bytes()), original_result_sha256=sha((native / "report.json").read_bytes()),
                models=actual_rows, original_saved_probabilities_audited=8192, audit_model_invocations=0,
                auditor_sha256=sha(Path(__file__).read_bytes()),
                limits="Independent struct/binary64 reductions of saved original native bytes, both repeats and all startup, against untouched GPU references. Original executable/compile/execution/packet identities checked. No re-inference; does not reproduce GPU TF/H5 or establish recognition, fullstream/phase qualification, P4 resource/timing or acceptance.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--native-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.packet_root, a.native_root, a.plan)
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
