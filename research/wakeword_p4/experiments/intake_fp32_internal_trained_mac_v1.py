"""Authenticate the original GPU software packet before extracting; no inference."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

import numpy as np

from inspect_fp32_internal_model_mac_v2 import inspect

PREFIX = "research/wakeword_p4/"
ARCHIVE_SHA = "06c53a9334a59ff07ce21e85afad9e9bf29fb79da2fe4879be3e560b0b06260a"
INPUT_SHA = "0f7eff41940be4149be682f9a24755d42646bbe08e211c5fbdd1638059d3c9ac"
H5 = {
    "hey_willow-float_control": "6a763ffd0dd6c4d9a3c43395602a1b106baf8723328487b613855bc8bacfb829",
    "hey_willow-integer_qat": "bc6b59109ff85ae4cb91b94690bf7d26b63438281082bcb296e610e864956f54",
    "hey_nyx-float_control": "a8e45f24c959a31d8ea4c303b5413295873041dc7d13702182e528bf6f06dd77",
    "hey_nyx-integer_qat": "dc3a123d1467e97f9057f665184a8af43b908d49f39f8f31abe0f9ce7ef2a5b9",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, commit, name):
    return subprocess.check_output(["git", "show", commit + ":" + PREFIX + name], cwd=repo)


def intake(repo, commit, archive, output):
    raw = archive.read_bytes()
    assert len(raw) == 527033 and sha(raw) == ARCHIVE_SHA
    committed_manifest = git_bytes(repo, commit, "results/gpu-fp32-internal-trained-mac-v1-manifest.json")
    manifest = json.loads(committed_manifest)
    with tarfile.open(fileobj=BytesIO(raw), mode="r:gz") as tar:
        members = tar.getmembers()
        expected = set(manifest["files"]) | {"manifest.json"}
        assert len(members) == len(expected) == 19
        assert {m.name for m in members} == expected
        payload = {}
        for member in members:
            p = PurePosixPath(member.name)
            assert member.isfile() and not p.is_absolute() and ".." not in p.parts
            assert member.name == p.as_posix() and member.size <= 491520
            payload[member.name] = tar.extractfile(member).read()
        assert payload["manifest.json"] == committed_manifest
    for name, row in manifest["files"].items():
        assert len(payload[name]) == row["bytes"] and sha(payload[name]) == row["sha256"], name
    for packed, public in [
        ("receipts/result.json", "results/4090-fp32-internal-software-original-result-v1.json"),
        ("receipts/4090-fp32-internal-software-execution-v1.json", "results/4090-fp32-internal-software-execution-v1.json"),
        ("receipts/run_fp32_internal_software_v1.py", "experiments/run_fp32_internal_software_v1.py"),
        ("receipts/package_fp32_internal_mac_v1.py", "experiments/package_fp32_internal_mac_v1.py"),
    ]:
        assert payload[packed] == git_bytes(repo, commit, public), packed
    lock = json.loads(payload["receipts/4090-fp32-internal-software-execution-v1.json"])
    result = json.loads(payload["receipts/result.json"])
    assert result["execution_sha256"] == sha(payload["receipts/4090-fp32-internal-software-execution-v1.json"])
    assert result["all_passed"] and len(result["cases"]) == 4
    assert {m["target"] + "-" + m["condition"]: m["sha256"] for m in lock["models"]} == H5
    assert sha(payload["inputs.bin"]) == INPUT_SHA
    source_name = "receipts/run_fp32_internal_software_v1.py"
    assert sha(payload[source_name]) in lock["files"].values()
    references, contracts = {}, {}
    for name, h5 in H5.items():
        row = json.loads(payload[name + "/receipt.json"])
        assert row == next(r for r in result["cases"] if r["target"] + "-" + r["condition"] == name)
        assert row["h5_sha256"] == h5 and row["parameters_unchanged"] and row["passed"]
        assert row["model_sha256"] == sha(payload[name + "/model.tflite"])
        assert row["references_sha256"] == sha(payload[name + "/references.npz"])
        contracts[name] = inspect(payload[name + "/model.tflite"])
        with np.load(BytesIO(payload[name + "/references.npz"]), allow_pickle=False) as z:
            assert set(z.files) == {"tf", "litert"}
            arrays = [z[k] for k in ("tf", "litert")]
            assert all(a.dtype == np.float32 and a.shape == (4, 2, 256) and np.isfinite(a).all() for a in arrays)
            assert all(a[:, 0].tobytes() == a[:, 1].tobytes() for a in arrays)
            gap = float(np.max(np.abs(arrays[0].astype(np.float64) - arrays[1].astype(np.float64))))
            assert gap == row["max_abs"] and gap <= 1e-4
            references[name] = dict(probabilities_per_backend=2048, tf_litert_max_abs=gap,
                                    fresh_repeats_byte_exact=True)
    # Nothing is materialized until all identities, references and contracts pass.
    output.mkdir(exist_ok=False)
    for name, data in payload.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as f:
            f.write(data)
    return dict(schema="mac-fp32-internal-trained-intake-v1", gpu_commit=commit,
                archive_bytes=len(raw), archive_sha256=sha(raw), manifest_sha256=sha(committed_manifest),
                payloads_verified=18, exact_members=19, original_sources_and_receipts_exact=True,
                h5_bindings=H5, contracts=contracts, references=references,
                inspector_sha256=sha(Path(__file__).with_name("inspect_fp32_internal_model_mac_v2.py").read_bytes()),
                intake_source_sha256=sha(Path(__file__).read_bytes()), model_invocations=0, media_reads=0,
                limits="Actual models/FlatBuffers and saved references independently inspected. Original H5 bytes are not in this packet; H5 equivalence is a receipt binding, not a Mac H5 or TF reproduction. No inference, recognition, FAR or P4 evidence.")


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
    print(json.dumps(dict(report_sha256=sha(a.report.read_bytes()), models=len(report["contracts"]), model_invocations=0)))


if __name__ == "__main__":
    main()
