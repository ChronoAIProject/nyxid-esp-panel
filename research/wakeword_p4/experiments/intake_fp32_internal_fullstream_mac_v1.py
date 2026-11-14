"""Authenticate original GPU fullstream captures and independently audit saved arrays."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np

from inspect_fp32_internal_model_mac_v2 import inspect
from intake_fp32_internal_trained_mac_v1 import H5

PREFIX = "research/wakeword_p4/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit_arrays(data, samples):
    with np.load(BytesIO(data), allow_pickle=False) as z:
        assert set(z.files) == {"raw", "input", "tf", "tf_repeat", "litert", "litert_repeat",
                               "offline_phase", "offline_nominal", "support_unclamped", "sample_end"}
        raw, values = z["raw"], z["input"]
        assert raw.dtype == np.uint16 and raw.ndim == 2 and raw.shape[1] == 40 and len(raw) > 0
        padded = np.pad(raw, ((0, (-len(raw)) % 3), (0, 0))).astype(np.float32)
        expected = np.clip(padded / 25.6, 0, 26)
        assert values.dtype == np.float32 and values.shape == expected.shape and values.tobytes() == expected.tobytes()
        steps = len(values) // 3
        assert steps >= 75
        for name in ["tf", "tf_repeat", "litert", "litert_repeat", "offline_phase", "offline_nominal"]:
            a = z[name]
            assert a.dtype == np.float32 and a.shape == ((steps - 74,) if name.startswith("offline") else (steps,))
            assert np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
        for backend in ["tf", "litert"]:
            assert z[backend].tobytes() == z[backend + "_repeat"].tobytes()
        clock = np.array([(3 * (i + 1) + 2) * 160 for i in range(steps)], dtype=np.int64)
        assert np.array_equal(z["support_unclamped"], clock)
        assert np.array_equal(z["sample_end"], np.minimum(clock, samples))
        gap = float(np.max(np.abs(z["tf"].astype(np.float64) - z["litert"].astype(np.float64))))
        phase = float(np.max(np.abs(z["offline_phase"].astype(np.float64) - z["tf"][74:].astype(np.float64))))
        identities = {name: dict(dtype=str(z[name].dtype), shape=list(z[name].shape), sha256=sha(z[name].tobytes()))
                      for name in ["raw", "input", "support_unclamped", "sample_end"]}
        return dict(steps=steps, max_abs=gap, phase_max_abs=phase, passed=max(gap, phase) <= 1e-4,
                    identity=identities, input_sha256=sha(values.astype("<f4").tobytes()))


def intake(repo, commit, archive, output):
    def public(name):
        return subprocess.check_output(["git", "show", commit + ":" + PREFIX + name], cwd=repo)

    manifest_bytes = public("results/gpu-fp32-internal-fullstream-mac-v1-manifest.json")
    manifest = json.loads(manifest_bytes)
    download = json.loads(public("results/gpu-fp32-internal-fullstream-mac-v1-download.json"))
    raw = archive.read_bytes()
    assert len(raw) == download["bytes"] and sha(raw) == download["sha256"]
    assert download["all_passed"] and download["verified_payloads"] == 556
    assert manifest["cases"] == 540 and manifest["requests"] == 135 and manifest["all_numerical_passed"]
    assert manifest["source_commit"] == download["source_commit"]
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as tar:
        members = tar.getmembers()
        assert len(members) == 557 and {m.name for m in members} == set(manifest["files"]) | {"manifest.json"}
        assert sum(m.size for m in members) < 150 * 1024 * 1024
        payload = {}
        for member in members:
            p = PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and ".." not in p.parts and member.name == p.as_posix()
            assert member.size <= 2 * 1024 * 1024
            payload[member.name] = tar.extractfile(member).read()
        assert payload["manifest.json"] == manifest_bytes
    for name, row in manifest["files"].items():
        assert len(payload[name]) == row["bytes"] and sha(payload[name]) == row["sha256"], name
    for name in ["4090-fp32-internal-fullstream-execution-v1.json", "4090-fp32-internal-fullstream-original-result-v1.json",
                 "4090-fp32-fullstream-independent-arrays-v1.json", "4090-fp32-fullstream-original-frontend-audit-v1.json",
                 "4090-mac-fp32-trained-capture-intake-v1.json", "4090-mac-fp32-trained-original-result-v1.json"]:
        assert payload["receipts/" + name] == public("results/" + name), name
    for name in ["package_fp32_fullstream_mac_v1.py", "run_fp32_internal_fullstream_v1.py",
                 "audit_fp32_fullstream_arrays_v1.py", "audit_fp32_fullstream_frontend_v1.py"]:
        assert payload["receipts/" + name] == public("experiments/" + name), name
    original_mac = repo / (PREFIX + "results/mac-fp32-trained-real-tflm-host-result-v1.json")
    assert payload["receipts/4090-mac-fp32-trained-original-result-v1.json"] == original_mac.read_bytes()
    lock_name = "receipts/4090-fp32-internal-fullstream-execution-v1.json"
    lock = json.loads(payload[lock_name])
    result = json.loads(payload["receipts/result.json"])
    gpu_audit = json.loads(payload["receipts/4090-fp32-fullstream-independent-arrays-v1.json"])
    frontend = json.loads(payload["receipts/4090-fp32-fullstream-original-frontend-audit-v1.json"])
    assert payload["receipts/result.json"] == payload["receipts/4090-fp32-internal-fullstream-original-result-v1.json"]
    assert [json.loads(line) for line in payload["receipts/cases.jsonl"].splitlines()] == result["cases"]
    assert result["execution_sha256"] == frontend["execution_sha256"] == sha(payload[lock_name])
    assert result["all_passed"] and frontend["all_passed"] and gpu_audit["all_passed"]
    assert len(result["cases"]) == len(gpu_audit["cases"]) == 540 and len(frontend["rows"]) == 135
    prospective = json.loads(public("results/4090-fp32-internal-fullstream-prospective-v1.json"))
    assert lock["requests"] == prospective["ordered_requests"] and len(lock["requests"]) == 135
    assert lock["max_abs"] == 1e-4 and lock["retries"] == 0
    assert {m["target"] + "-" + m["condition"]: m["sha256"] for m in lock["models"]} == H5
    prior = json.loads((repo / (PREFIX + "results/mac-fp32-internal-trained-intake-v1.json")).read_text())
    contracts = {}
    for export in lock["exports"]:
        name = export["target"] + "-" + export["condition"]
        data = payload["models/" + name + ".tflite"]
        assert sha(data) == export["sha256"] == prior["contracts"][name]["model_sha256"]
        contracts[name] = inspect(data)
    assert list(contracts) == list(H5)
    requests, cases, shared = lock["requests"], [], {}
    expected_order = [(name, index) for name in H5 for index in range(135)]
    assert [(r["target"] + "-" + r["condition"], r["index"]) for r in result["cases"]] == expected_order
    for row, gpu_row in zip(result["cases"], gpu_audit["cases"], strict=True):
        name = row["target"] + "-" + row["condition"]
        index = row["index"]
        member = "arrays/" + name + f"-{index:03}.npz"
        assert sha(payload[member]) == row["sha256"] and row["request_order"] == requests[index]["order"]
        checked = audit_arrays(payload[member], requests[index]["samples"])
        assert all(checked[k] == row[k] == gpu_row[k] for k in ["max_abs", "phase_max_abs", "passed"])
        assert gpu_row["key"] == [row["target"], row["condition"], index]
        assert checked["passed"] and row["finite"] and row["repeat_exact"]
        if index in shared:
            assert checked["identity"] == shared[index]["identity"]
        else:
            shared[index] = checked
        original_raw = frontend["rows"][index]
        assert original_raw["index"] == index and original_raw["request_order"] == requests[index]["order"]
        assert original_raw["raw_sha256"] == checked["identity"]["raw"]["sha256"]
        assert original_raw["shape"] == checked["identity"]["raw"]["shape"] and original_raw["models_matched"] == 4
        cases.append(dict(model=name, index=index, request_order=row["request_order"], member=member,
                          original_npz_sha256=row["sha256"], steps=checked["steps"], input_sha256=checked["input_sha256"],
                          original_tf_litert_max_abs=checked["max_abs"], original_phase_max_abs=checked["phase_max_abs"]))
    output.mkdir(exist_ok=False)
    for name, data in payload.items():
        dest = output / name
        dest.parent.mkdir(exist_ok=True, parents=True)
        with dest.open("xb") as f:
            f.write(data)
    return dict(schema="mac-fp32-internal-fullstream-intake-v1", gpu_commit=commit, archive_bytes=len(raw),
                archive_sha256=sha(raw), manifest_sha256=sha(manifest_bytes), exact_members=557, payloads_verified=556,
                original_receipts_sources_and_mac_result_exact=True, contracts=contracts, cases=cases,
                unique_input_streams=len(shared), original_reference_steps=sum(c["steps"] for c in cases),
                expected_native_invocations=sum(c["steps"] for c in cases) * 2,
                input_and_clock_identity_shared_across_four_models=True, model_invocations=0, audio_decodes=0,
                intake_source_sha256=sha(Path(__file__).read_bytes()), all_passed=True,
                limits="Original all540 saved fullstream arrays/model/source/receipt bytes authenticated and independently reduced, including startup, padding, input scaling, clocks, repeat and phase gates. Original PCM is not included; Mac did not reproduce frontend or GPU H5/TF. No native fullstream inference, recognition/FAR, P4 or fresh development/acceptance qualification.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--gpu-commit", required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    report = intake(a.repository_root, a.gpu_commit, a.archive, a.output_dir)
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["archive_sha256", "payloads_verified", "original_reference_steps", "expected_native_invocations", "all_passed"]}))


if __name__ == "__main__":
    main()
