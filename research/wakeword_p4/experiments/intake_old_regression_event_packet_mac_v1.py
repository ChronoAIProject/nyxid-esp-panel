"""Authenticate original event-only bytes, then materialize the frozen reader tree."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

PREFIX = "research/wakeword_p4/"
RESULT_REF = "ca79386ea4082036bae9875a2cfdccb8394f8b6d"
PACKET_REF = "4701ce39f6ee5d072c5678a464750023b2cfdc5a"
ARCHIVE_SHA = "69308a8be5140770abaa31e214821e3296702bd9c345eb39c10efc52c5b2fd78"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--isolated-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()

    def tracked(ref, path):
        return subprocess.check_output(["git", "-C", str(a.repository_root), "show", ref + ":" + path])

    receipt_names = [PREFIX + "results/" + name + ".json" for name in [
        "4090-old-fixed-grid-regression-plan-v2", "4090-old-fixed-grid-regression-collection-v2",
        "4090-old-fixed-grid-regression-metrics-v2", "4090-old-fixed-grid-regression-independent-audit-v2",
        "4090-old-regression-audit-continuation-result-v1"]]
    receipt_bytes = {name: tracked(RESULT_REF, name) for name in receipt_names}
    plan = json.loads(receipt_bytes[receipt_names[0]])
    metrics = json.loads(receipt_bytes[receipt_names[2]])
    required = {plan["synthetic_manifest"]: plan["files"][plan["synthetic_manifest"]]}
    required.update({row["events_path"]: row["events_sha256"] for row in metrics["points"]})
    assert len(metrics["points"]) == 1242 and len(required) == 1243
    archive_bytes = a.archive.read_bytes()
    assert len(archive_bytes) == 527610 and sha(archive_bytes) == ARCHIVE_SHA
    published_manifest = tracked(PACKET_REF, PREFIX + "results/gpu-old-regression-event-ledgers-mac-v1-manifest.json")
    expected = required | {path: sha(data) for path, data in receipt_bytes.items()}
    payload = {}
    with tarfile.open(a.archive, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == 1249
        assert set(names) == set(expected) | {"manifest.json"}
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute() and
                   ".." not in PurePosixPath(m.name).parts for m in members)
        manifest_bytes = archive.extractfile("manifest.json").read()
        assert manifest_bytes == published_manifest
        manifest = json.loads(manifest_bytes)
        entries = {row["path"]: row for row in manifest["files"]}
        assert len(entries) == len(manifest["files"]) == 1248
        assert set(entries) == set(expected)
        for path, digest in expected.items():
            data = archive.extractfile(path).read()
            assert sha(data) == entries[path]["sha256"] == digest, path
            assert len(data) == entries[path]["size"], path
            payload[path] = data
    root = a.isolated_root.resolve()
    assert subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip().startswith("9162bdb")
    for path, data in payload.items():
        destination = root / path
        assert destination.resolve().is_relative_to(root)
        if destination.exists():
            assert destination.read_bytes() == data, path
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as f:
                f.write(data)
    report = dict(schema="mac-old-regression-event-packet-intake-v1", script_sha256=sha(Path(__file__).read_bytes()),
                  packet_ref=PACKET_REF, original_results_ref=RESULT_REF,
                  archive_bytes=len(archive_bytes), archive_sha256=ARCHIVE_SHA,
                  manifest_sha256=sha(manifest_bytes), published_manifest_byte_exact=True,
                  exact_archive_members=1249, original_private_payloads=1243, original_event_npz=1242,
                  original_receipts=5, all_original_payload_and_receipt_hashes_exact=True,
                  required_payload_hashes_sha256=sha(json.dumps(required, sort_keys=True).encode()),
                  isolated_head="9162bdb", frozen_reader_changed=False,
                  models_or_scores_or_audio_opened=False, fresh_development_or_acceptance_accessed=False,
                  limits="Authenticates original consumed-regression bytes only. No inference, new events, threshold selection, recognition qualification or promotion.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
