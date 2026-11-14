"""Publish original complete native fullstream captures for independent cross-audit."""
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
    p.add_argument("--build-root", type=Path, required=True)
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
    for name in ["build-plan.json", "compile.json", "esp_timer.h"]:
        payload["runtime/" + name] = (a.build_root / name).read_bytes()
    for name in ["intake_fp32_internal_fullstream_mac_v1.py", "run_fp32_internal_fullstream_host_mac_v1.py",
                 "audit_fp32_internal_fullstream_host_mac_v1.py", "fp32_internal_fullstream_host_v1.cpp",
                 "build_fp32_internal_fullstream_host_mac_v1.py", "inspect_fp32_internal_model_mac_v2.py",
                 "intake_fp32_internal_trained_mac_v1.py", "run_fp32_real_tflm_host_mac_v1.py",
                 "fp32_internal_fullstream_host_mac_v1.md", Path(__file__).name]:
        payload["sources/" + name] = (root / "experiments" / name).read_bytes()
    for name in ["mac-fp32-internal-fullstream-intake-v1.json", "mac-fp32-internal-fullstream-host-execution-v1.json",
                 "mac-fp32-internal-fullstream-host-result-v1.json", "mac-fp32-internal-fullstream-host-saved-byte-audit-v1.json",
                 "mac-fp32-internal-fullstream-host-build-v1.json", "mac-fp32-internal-fullstream-host-build-plan-v1.json",
                 "mac-fp32-internal-fullstream-host-compile-v1.json"]:
        payload["reports/" + name] = (root / "results" / name).read_bytes()
    assert payload["native/report.json"] == payload["reports/mac-fp32-internal-fullstream-host-result-v1.json"]
    audit = json.loads(payload["reports/mac-fp32-internal-fullstream-host-saved-byte-audit-v1.json"])
    assert audit["all_passed"] and audit["original_probabilities_audited"] == 192336
    assert len([n for n in payload if n.startswith("native/") and n.endswith("/outputs.bin")]) == 540
    manifest = dict(schema="mac-fp32-internal-fullstream-host-handoff-v1",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                    files={name: dict(bytes=len(data), sha256=sha(data)) for name, data in sorted(payload.items())},
                    original_cases=540, actual_native_invocations=192336, independent_saved_byte_audit_passed=True,
                    limits="Original host64-bit fullstream numerical captures and untouched GPU packet, native inputs/output/model/executable/stdout/receipts, prospective locks and audit sources. No audio/H5 or credentials. No P4 measurement, fresh development, recognition/FAR/acceptance qualification or candidate promotion.")
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    with a.manifest.open("xb") as f:
        f.write(manifest_bytes)
    payload["manifest.json"] = manifest_bytes
    with a.archive.open("xb") as out:
        with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, data in sorted(payload.items()):
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, BytesIO(data))
    with tarfile.open(a.archive, "r:gz") as archive:
        assert len(archive.getmembers()) == len(payload)
        assert all(m.isfile() and archive.extractfile(m).read() == payload[m.name] for m in archive.getmembers())
    print(json.dumps(dict(archive_bytes=a.archive.stat().st_size, archive_sha256=sha(a.archive.read_bytes()),
                         exact_members=len(payload), verified_payloads=len(payload) - 1)))


if __name__ == "__main__":
    main()
