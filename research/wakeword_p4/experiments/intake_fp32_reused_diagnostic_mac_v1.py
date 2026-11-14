"""Authenticate the GPU diagnostic packet and create an exact-byte relocated view."""
import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

BASE = "research/wakeword_p4/"
OUT = ".local/wakeword/fp32-reused-development-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def intake(repo, commit, archive, output):
    def public(name, ref=commit):
        return subprocess.check_output(["git", "show", ref + ":" + BASE + name], cwd=repo)

    manifest_bytes = public("results/gpu-fp32-reused-development-diagnostic-mac-v1-manifest.json")
    manifest = json.loads(manifest_bytes)
    assert manifest["schema"] == "fp32-reused-development-diagnostic-mac-v1"
    assert manifest["original_traces"] == 1488 and manifest["points"] == 552
    subprocess.run(["git", "merge-base", "--is-ancestor", manifest["source_commit"], commit], cwd=repo, check=True)
    original = archive.read_bytes()
    assert len(original) < 256 * 1024 * 1024
    with tarfile.open(fileobj=BytesIO(original), mode="r:gz") as t:
        members = t.getmembers()
        assert len(members) == len(manifest["files"]) + 1
        assert {m.name for m in members} == set(manifest["files"]) | {"manifest.json"}
        assert sum(m.size for m in members) < 512 * 1024 * 1024
        payload = {}
        for m in members:
            path = PurePosixPath(m.name)
            assert m.isfile() and not path.is_absolute() and ".." not in path.parts and path.as_posix() == m.name
            assert m.size < 128 * 1024 * 1024
            payload[m.name] = t.extractfile(m).read()
    assert payload["manifest.json"] == manifest_bytes
    for name, identity in manifest["files"].items():
        assert len(payload[name]) == identity["bytes"] and sha(payload[name]) == identity["sha256"], name
    for name in ["4090-fp32-reused-development-execution-v1.json", "4090-fp32-reused-collection-independent-audit-v1.json",
                 "4090-fp32-reused-scoring-execution-v1.json", "4090-fp32-reused-scoring-independent-audit-v1.json",
                 "4090-fp32-reused-scoring-descriptive-v1.json", "4090-original-official-development-calibration-v1.json",
                 "4090-hey-official-development-execution-v1.json"]:
        assert payload["receipts/" + name] == public("results/" + name), name
    assert payload["receipts/result.json"] == public("results/4090-fp32-reused-collection-original-result-v1.json")
    assert payload["receipts/summary.json"] == public("results/4090-fp32-reused-scoring-original-summary-v1.json")
    for name in manifest["files"]:
        if name.startswith("source/"):
            assert payload[name] == public("experiments/" + name.removeprefix("source/"), manifest["source_commit"])
    lock = json.loads(payload["receipts/4090-fp32-reused-development-execution-v1.json"])
    result = json.loads(payload["receipts/result.json"])
    score_audit = json.loads(payload["receipts/4090-fp32-reused-scoring-independent-audit-v1.json"])
    assert result["all_passed"] and result["execution_sha256"] == sha(payload["receipts/4090-fp32-reused-development-execution-v1.json"])
    assert sha(payload["receipts/summary.json"]) == score_audit["summary_sha256"]
    relocation = {OUT + "/result.json": "receipts/result.json", OUT + "/diagnostic/summary.json": "receipts/summary.json"}
    for phase, material in lock["materials"].items():
        source = "materials/" + phase + ".json"
        assert sha(payload[source]) == material["sha256"]
        relocation[material["path"]] = source
    traces = 0
    for model in lock["models"]:
        name = model["target"] + "-" + model["condition"]
        report_member = "reports/" + name + ".json"
        assert sha(payload["models/" + name + ".tflite"]) == model["sha256"]
        relocation[model["path"]] = "models/" + name + ".tflite"
        assert sha(payload[report_member]) == next(m["report_sha256"] for m in result["models"] if m["model"] == name)
        relocation[OUT + "/" + name + "/report.json"] = report_member
        relocation[OUT + "/" + name + "/records.jsonl"] = "reports/" + name + "-records.jsonl"
        relocation[OUT + "/diagnostic/" + name + ".jsonl"] = "diagnostic/" + name + ".jsonl"
        for rec in json.loads(payload[report_member])["records"]:
            member = f"traces/{name}/{rec['phase']}/{rec['index']:06}.npz"
            assert sha(payload[member]) == rec["trace_sha256"]
            assert rec["trace_path"] == f"{OUT}/{name}/{rec['phase']}/{rec['index']:06}.npz"
            relocation[rec["trace_path"]] = member
            traces += 1
    assert traces == 1488
    output.mkdir(exist_ok=False)
    for prefix, items in [("packet", {k: k for k in payload}), ("capture", relocation)]:
        for name, member in items.items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and path.as_posix() == name
            dest = output / prefix / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(payload[member])
    return dict(schema="mac-fp32-reused-diagnostic-intake-v1", gpu_commit=commit,
                packet_source_commit=manifest["source_commit"], archive_bytes=len(original), archive_sha256=sha(original),
                manifest_sha256=sha(manifest_bytes), exact_members=len(payload), payloads_verified=len(manifest["files"]),
                original_traces=traces, relocated_original_files=len(relocation), relocation_preserves_all_bytes=True,
                all_passed=True, model_invocations=0, media_reads=0, thresholds_selected=False,
                intake_source_sha256=sha(Path(__file__).read_bytes()),
                limits="Original packet membership/content authenticated to published git manifest and original public locks/receipts. Relocation copies bytes only. Actual probability/event/metric audit is a separate step; no recognition or P4 qualification.")


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
    print(json.dumps(report))


if __name__ == "__main__":
    main()
