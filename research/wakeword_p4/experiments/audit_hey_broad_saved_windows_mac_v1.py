"""Independently reduce authenticated saved coordinates against original metadata.

No sampler import, RNG calls, tensor reconstruction, model or audio access.
The remote coordinate-to-tensor audit is explicitly separate evidence.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


PIN = "e1f341a9e2dc8dd073efd1f50e79d4912edbfed5"
BASE = "research/wakeword_p4"
MANIFEST_SHA = "97e73ba9aade7d3e351f68225848ba2c3003d773a6588678d24a6671e41d07bb"
TARGETS = ("hey_willow", "hey_nyx")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, path):
    return subprocess.check_output(["git", "show", f"{PIN}:{BASE}/{path}"], cwd=repo)


def authenticate_packet(packet):
    manifest_bytes = (packet / "manifest.json").read_bytes()
    assert sha(manifest_bytes) == MANIFEST_SHA
    manifest = json.loads(manifest_bytes)
    expected = {"manifest.json"}
    for entry in manifest["payloads"]:
        name = entry["path"]
        assert name not in expected and not Path(name).is_absolute() and ".." not in Path(name).parts
        expected.add(name)
        data = (packet / name).read_bytes()
        assert len(data) == entry["bytes"] and sha(data) == entry["sha256"], name
    assert {str(p.relative_to(packet)) for p in packet.rglob("*") if p.is_file()} == expected
    assert len(expected) == 13


def load_rows(repo, metadata, execution):
    original = json.loads(git_bytes(repo, "results/4090-hey-broad-features-v1.json"))
    rows, receipts = {}, {}
    for target in TARGETS:
        root = metadata / target
        provenance_bytes = (root / "target-provenance.json").read_bytes()
        provenance = json.loads(provenance_bytes)
        assert provenance == original["targets"][target]
        data = (root / "features.json").read_bytes()
        assert sha(data) == provenance["files"]["features.json"]
        for name, raw in (("features.json", data), ("target-provenance.json", provenance_bytes)):
            path = f".local/wakeword/hey-broad-training-features-v1/{target}/{name}"
            assert sha(raw) == execution["files"][path]
        records = json.loads(data)
        assert len(records) == 106956
        for row in records:
            assert row["split"] == "training" and row["frames"] >= 225
            assert row["label"] == int(row["target_id"] == target)
            assert row["engine"] == row["source_pair_key"]["engine"]
        rows[target] = records
        receipts[target] = dict(metadata_sha256=sha(data), provenance_sha256=sha(provenance_bytes))
    return rows, receipts


def row_role(row, target):
    if row["target_id"] == target:
        return "own"
    return "common" if row["target_id"] == "common_negative" else "reciprocal"


def validate_coordinate(rec, rows, target, discarded=False):
    assert all(type(rec[k]) is int for k in ("start", "end", "label"))
    start, end = rec["start"], rec["end"]
    assert start >= 0 and end - start == 225 and rec["label"] in (0, 1)
    if rec["kind"] == "ambient":
        assert not discarded and rec["label"] == 0
        assert type(rec["ambient_index"]) is int and rec["ambient_index"] >= 0
        return None
    assert rec["kind"] in ("ordinary", "prefix")
    assert type(rec["row"]) is int and 0 <= rec["row"] < len(rows)
    row = rows[rec["row"]]
    assert type(rec["offset"]) is int and rec["offset"] == row["offset"]
    assert end <= row["frames"]
    if discarded:
        assert rec["kind"] == "ordinary" and rec["label"] == row["label"] == 0
    if rec["kind"] == "prefix":
        assert rec["label"] == 0 and row["label"] == 1
        assert row_role(row, target) == "own" and row["engine"] in ("piper", "kokoro")
        assert 15 <= row["keyword_frame"] - end <= 35
    else:
        assert rec["label"] == row["label"]
        if rec["label"]:
            # The original sampler clamps the proposed positive endpoint.
            lo = min(row["frames"], max(225, row["keyword_frame"] + 3))
            hi = min(row["frames"], max(225, row["keyword_frame"] + 27))
        else:
            lo, hi = max(225, row["lead_frame"] - 25), row["frames"]
        assert lo <= end <= hi
    return row


class Reduction:
    def __init__(self, target, rows):
        self.target, self.rows = target, rows
        self.groups, self.sources, self.conditioning = Counter(), defaultdict(set), defaultdict(set)
        self.boundaries, self.carriers, self.slot_counts = Counter(), Counter(), Counter()
        self.discarded = 0
        self.eligible = hashlib.sha256()
        self.batch_hashes = hashlib.sha256()

    def batch(self, data, step):
        assert data["step"] == step and re.fullmatch(r"[0-9a-f]{64}", data["batch_sha256"])
        assert len(data["survivors"]) == 128 and len(data["overwritten_ordinary"]) == 16
        self.batch_hashes.update(bytes.fromhex(data["batch_sha256"]))
        for rec in data["overwritten_ordinary"]:
            validate_coordinate(rec, self.rows, self.target, discarded=True)
            self.discarded += 1
        slots, paired = Counter(), []
        for position, rec in enumerate(data["survivors"]):
            row = validate_coordinate(rec, self.rows, self.target)
            kind, label = rec["kind"], rec["label"]
            slots[kind, label] += 1
            if row is None:
                key = ("ambient", "background", "ambient", "none", 0, "")
                paired.append((rec, None))
            else:
                role = row_role(row, self.target)
                key = (kind, role, row["engine"], row["source_pair_key"]["carrier"], label, row["text"])
                self.sources[key].add(row["source_id"])
                self.conditioning[key].add(str(row["source_pair_key"]["conditioning"]))
                self.boundaries[kind, "windows"] += 1
                if row["keyword_frame"] is not None:
                    self.boundaries[kind, "contains_generated_prefix_boundary"] += int(
                        rec["start"] <= row["keyword_frame"] < rec["end"])
                self.carriers[kind, role, key[3]] += 1
                pair_key = (kind, role, row["engine"], row["source_pair_key"], row["variant"])
                paired.append((rec, pair_key))
                if kind == "ordinary" and role == "common" and row["engine"] == "kokoro":
                    identity = [step, position, rec["row"], rec["offset"], rec["start"], rec["end"]]
                    self.eligible.update((json.dumps(identity, separators=(",", ":")) + "\n").encode())
            self.groups[key] += 1
        assert slots == {("ordinary", 1): 32, ("ordinary", 0): 32, ("prefix", 0): 16, ("ambient", 0): 48}
        self.slot_counts.update(slots)
        return paired

    def finish(self, expected):
        grouped = [dict(slot=k[0], role=k[1], engine=k[2], carrier=k[3], label=k[4], text=k[5],
                        conditioning_ids=len(self.conditioning[k]), windows=v,
                        unique_source_containers=len(self.sources[k])) for k, v in sorted(self.groups.items())]
        boundaries = [dict(slot=k[0], measure=k[1], count=v) for k, v in sorted(self.boundaries.items())]
        assert grouped == expected["counts"]
        assert boundaries == expected["generated_boundary_coverage"]
        assert self.discarded == 160000 and sum(self.slot_counts.values()) == 1280000
        return dict(
            target=self.target, batches=10000, retained_windows=1280000,
            discarded_ordinary_windows=self.discarded, all_coordinates_valid=True,
            grouped_counts_exact_against_gpu=True, generated_boundary_counts_exact_against_gpu=True,
            counts=grouped, generated_boundary_coverage=boundaries,
            slot_counts=[dict(kind=k[0], label=k[1], windows=v) for k, v in sorted(self.slot_counts.items())],
            carrier_counts=[dict(kind=k[0], role=k[1], carrier=k[2], windows=v)
                            for k, v in sorted(self.carriers.items())],
            ordered_eligible_kokoro_coordinate_sha256=self.eligible.hexdigest(),
            ordered_reported_batch_sha256_digest=self.batch_hashes.hexdigest(),
        )


def run(repo, packet, metadata, output):
    assert not output.exists()
    authenticate_packet(packet)
    result_path = packet / "results/4090-hey-broad-surviving-window-census-v2.json"
    result = json.loads(result_path.read_bytes())
    execution_path = packet / "results/4090-hey-broad-surviving-window-census-execution-v2.json"
    assert sha(execution_path.read_bytes()) == result["execution_sha256"]
    execution = json.loads(execution_path.read_bytes())
    assert execution["batches_per_target"] == 10000 and execution["targets"] == list(TARGETS)
    for name, expected_sha in execution["files"].items():
        if "/experiments/" in name or name.startswith(f"{BASE}/training/"):
            rel = ("experiments/" + name.rsplit("/", 1)[1]) if "/experiments/" in name else name[len(BASE) + 1:]
            if rel == "experiments/census_hey_broad_surviving_windows_v2.py":
                actual = (packet / "source/census_hey_broad_surviving_windows_v2.py").read_bytes()
            else:
                actual = git_bytes(repo, rel)
            assert sha(actual) == expected_sha, name
    rows, metadata_receipts = load_rows(repo, metadata, execution)
    expected = {r["target"]: r for r in result["targets"]}
    reductions = {t: Reduction(t, rows[t]) for t in TARGETS}
    pairing = Counter()
    paths = [packet / f"traces/{t}.jsonl.gz" for t in TARGETS]
    for t, p in zip(TARGETS, paths, strict=True):
        assert sha(p.read_bytes()) == expected[t]["trace_sha256"]
    with gzip.open(paths[0], "rt") as a, gzip.open(paths[1], "rt") as b:
        number = 0
        for number, lines in enumerate(zip(a, b, strict=True), 1):
            batches = [json.loads(line) for line in lines]
            paired = [reductions[t].batch(data, number) for t, data in zip(TARGETS, batches, strict=True)]
            for (ra, ka), (rb, kb) in zip(*paired, strict=True):
                pairing["survivor_positions"] += 1
                if ra["kind"] == rb["kind"] == "ambient":
                    pairing["ambient_positions"] += 1
                    pairing["ambient_coordinate_differences"] += int(ra != rb)
                else:
                    pairing["source_pair_key_differences"] += int(ka != kb)
                    if ka and ka[1] == "common":
                        pairing["common_positions"] += 1
                        pairing["common_coordinate_or_source_differences"] += int(ra != rb or ka != kb)
                        if ka[2] == "kokoro":
                            pairing["eligible_kokoro_common_positions"] += 1
                            pairing["eligible_kokoro_coordinate_differences"] += int(ra != rb or ka != kb)
            for ra, rb in zip(batches[0]["overwritten_ordinary"], batches[1]["overwritten_ordinary"], strict=True):
                pairing["discarded_positions"] += 1
                xa, xb = (rows[t][r["row"]] for t, r in zip(TARGETS, (ra, rb), strict=True))
                sa = (row_role(xa, TARGETS[0]), xa["engine"], xa["source_pair_key"], xa["variant"])
                sb = (row_role(xb, TARGETS[1]), xb["engine"], xb["source_pair_key"], xb["variant"])
                pairing["discarded_source_pair_key_differences"] += int(sa != sb)
            if number % 1000 == 0:
                print(json.dumps(dict(completed_paired_batches=number)), flush=True)
        assert number == 10000
    independent_path = packet / "results/4090-hey-broad-surviving-window-independent-audit-v1.json"
    remote = json.loads(independent_path.read_bytes())
    assert remote["all_passed"] and remote["original_result_sha256"] == sha(result_path.read_bytes())
    assert remote["source_sha256"] == sha((packet / "source/audit_hey_broad_surviving_window_trace_v1.py").read_bytes())
    report = dict(
        schema="mac-original-broad-saved-window-independent-metadata-audit-v1",
        source_sha256=sha(Path(__file__).read_bytes()), original_source_commit=PIN,
        packet_manifest_sha256=MANIFEST_SHA, original_gpu_result_sha256=sha(result_path.read_bytes()),
        original_gpu_execution_sha256=sha(execution_path.read_bytes()),
        original_metadata=metadata_receipts,
        targets=[reductions[t].finish(expected[t]) for t in TARGETS],
        cross_target_shuffled_slot_comparison=dict(sorted(pairing.items())),
        remote_coordinate_to_tensor_evidence=dict(
            result_sha256=sha(independent_path.read_bytes()), source_sha256=remote["source_sha256"],
            all_passed_reported=True, mac_reconstructed_tensors=False),
        model_invocations=0,
        limits=[
            "Local metadata/coordinate validation, not tensor reconstruction or a replay of sampler RNG.",
            "Reported batch hashes are authenticated saved values, not locally recomputed from arrays/schedules.",
            "Ambient index is checked for nonnegative integer and crop length only; source length/list unavailable locally.",
            "First10000 reconstructed batches per word, not all120000 or original historical live provenance.",
            "Generated boundaries are metadata proxies, not verified pronunciation or actual phonetic boundaries.",
            "No new media/model/feature arrays/fresh development/acceptance access; not recognition/FAR or causal proof.",
        ],
    )
    with output.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(report=str(output), cross_target=report["cross_target_shuffled_slot_comparison"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository-root", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--original-metadata", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()
    run(args.repository_root, args.packet, args.original_metadata, args.report)
