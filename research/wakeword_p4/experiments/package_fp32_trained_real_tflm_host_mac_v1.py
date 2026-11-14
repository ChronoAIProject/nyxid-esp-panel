"""Package original trained native execution, packet and independent audit bytes."""
import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--private-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    repo = a.repository_root.resolve()
    root = repo / "research/wakeword_p4"
    payload = {}
    for label in ["packet", "native"]:
        for path in sorted((a.private_root / label).rglob("*")):
            if path.is_file():
                assert not path.is_symlink()
                payload[label + "/" + str(path.relative_to(a.private_root / label))] = path.read_bytes()
    source_names = ["inspect_fp32_internal_model_mac_v2.py", "intake_fp32_internal_trained_mac_v1.py",
                    "review_fp32_internal_trained_contract_mac_v1.py", "run_fp32_real_tflm_host_mac_v1.py",
                    "run_fp32_trained_real_tflm_host_mac_v1.py", "fp32_trained_real_tflm_host_v1.cpp",
                    "audit_fp32_trained_real_tflm_host_mac_v1.py", Path(__file__).name,
                    "fp32_trained_real_tflm_host_mac_v1.md"]
    for name in source_names:
        payload["sources/" + name] = (root / "experiments" / name).read_bytes()
    reports = ["mac-fp32-internal-trained-initial-intake-failure-v1.json",
               "mac-fp32-internal-trained-contract-review-setup-failure-v1.json",
               "mac-fp32-internal-trained-contract-review-v1.json", "mac-fp32-internal-trained-intake-v1.json",
               "mac-fp32-trained-real-tflm-host-plan-v1.json", "mac-fp32-trained-real-tflm-host-result-v1.json",
               "mac-fp32-trained-real-tflm-host-saved-byte-audit-v1.json"]
    for name in reports:
        payload["reports/" + name] = (root / "results" / name).read_bytes()
    assert payload["native/report.json"] == payload["reports/mac-fp32-trained-real-tflm-host-result-v1.json"]
    audit = json.loads(payload["reports/mac-fp32-trained-real-tflm-host-saved-byte-audit-v1.json"])
    assert audit["all_passed"] and audit["original_saved_probabilities_audited"] == 8192
    manifest = dict(schema="mac-fp32-trained-real-tflm-host-handoff-v1",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    files={n: dict(bytes=len(b), sha256=sha(b)) for n, b in sorted(payload.items())},
                    actual_native_steps=8192, independent_saved_byte_audit_passed=True,
                    limits="Original host64-bit synthetic trained-model execution only. Original GPU models/references included unchanged; no H5/audio/development/acceptance or secrets. Not P4 memory/timing, fullstream/phase qualification, recognition or FAR. Prior static/fixture setup failures retained.")
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    with a.manifest.open("xb") as f:
        f.write(data)
    payload["manifest.json"] = data
    with a.archive.open("xb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, raw in sorted(payload.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(raw)
                    info.mode = 0o644
                    archive.addfile(info, BytesIO(raw))
    with tarfile.open(a.archive, "r:gz") as t:
        assert len(t.getmembers()) == len(payload)
        assert all(m.isfile() and t.extractfile(m).read() == payload[m.name] for m in t.getmembers())
    print(json.dumps(dict(archive_bytes=a.archive.stat().st_size, archive_sha256=sha(a.archive.read_bytes()),
                         exact_members=len(payload), payloads=len(payload) - 1, all_payloads_verified=True)))


if __name__ == "__main__":
    main()
