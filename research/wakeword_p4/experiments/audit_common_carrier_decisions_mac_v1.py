"""Independently reconstruct prospective slot decisions, not replacement tensors."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_hey_broad_saved_windows_mac_v1 import authenticate_packet, TARGETS


GPU_PIN = "32f77dc7ad0af1579503703cfab2f4b9ba40b8d1"
BASE = "research/wakeword_p4"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(repo, packet, metadata, prior_audit, output):
    assert not output.exists()
    authenticate_packet(packet)
    prior_bytes = prior_audit.read_bytes()
    prior = json.loads(prior_bytes)
    audit_source = Path(__file__).with_name("audit_hey_broad_saved_windows_mac_v1.py")
    assert prior["source_sha256"] == sha(audit_source.read_bytes())
    assert prior["schema"] == "mac-original-broad-saved-window-independent-metadata-audit-v1"
    assert all(t["all_coordinates_valid"] and t["grouped_counts_exact_against_gpu"] for t in prior["targets"])

    def read(path):
        return subprocess.check_output(["git", "show", f"{GPU_PIN}:{BASE}/{path}"], cwd=repo)

    decision_bytes = read("results/4090-hey-common-carrier-replacement-decisions-v1.json")
    expected = json.loads(decision_bytes)
    source = read("experiments/plan_common_carrier_replacements_v1.py")
    policy = read("experiments/common_carrier_slot_policy_v1.py")
    native_bytes = read("results/4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    native = json.loads(native_bytes)
    assert expected["source_sha256"] == sha(source)
    assert expected["policy_sha256"] == sha(policy)
    assert expected["native_execution_sha256"] == sha(native_bytes)
    assert expected["original_census_sha256"] == prior["original_gpu_result_sha256"]
    assert expected["original_independent_audit_sha256"] == prior["remote_coordinate_to_tensor_evidence"]["result_sha256"]
    assert np.__version__ == native["versions"]["numpy"] == "2.4.6"
    cells = {}
    for item in native["requests"]:
        request = item["request"]
        key = f'kokoro-negative-{request["anchor_order"]:05}', request["carrier"]
        assert key not in cells
        assert request["label"] == 0 and request["split"] == "training"
        assert request["target_id"] == "common_negative"
        cells[key] = request
    assert len(cells) == 1620

    results = []
    for target_expected in expected["targets"]:
        target = target_expected["target"]
        assert target in TARGETS
        raw = (metadata / target / "features.json").read_bytes()
        assert sha(raw) == prior["original_metadata"][target]["metadata_sha256"]
        rows = json.loads(raw)
        trace = packet / f"traces/{target}.jsonl.gz"
        assert sha(trace.read_bytes()) == target_expected["trace_sha256"]
        # Explicit PCG64 plus both integer draws, without importing GPU policy.
        rng = np.random.Generator(np.random.PCG64(90726))
        digest = hashlib.sha256()
        counts = Counter()
        samples = 0
        with gzip.open(trace, "rt") as f:
            for step, line in enumerate(f, 1):
                batch = json.loads(line)
                assert step == batch["step"]
                choices = []
                for slot, rec in enumerate(batch["survivors"]):
                    if rec["kind"] != "ordinary" or rec["label"] != 0:
                        continue
                    row = rows[rec["row"]]
                    if row["engine"] != "kokoro" or row["target_id"] != "common_negative":
                        continue
                    selected = bool(rng.integers(0, 2))
                    carrier = ("light", "time", "timer")[int(rng.integers(0, 3))]
                    req = cells[row["source_id"], carrier]
                    assert req["conditioning_id"] == row["source_pair_key"]["conditioning"]
                    assert req["speed"] == row["source_pair_key"]["scale"]
                    assert req["anchor_text"] == row["text"] and row["source_pair_key"]["carrier"] == "bare"
                    lower = max(225, row["lead_frame"] - 25)
                    span = row["frames"] - lower + 1
                    assert span > 0 and lower <= rec["end"] <= row["frames"]
                    choice = dict(slot=slot, replace=selected, carrier=carrier,
                                  anchor_id=row["source_id"], variant=row["variant"], native_order=req["order"],
                                  old_end=rec["end"], old_frames=row["frames"], old_lead=row["lead_frame"],
                                  crop_midpoint_numerator=2 * (rec["end"] - lower) + 1,
                                  crop_midpoint_denominator=2 * span)
                    assert 0 < choice["crop_midpoint_numerator"] < choice["crop_midpoint_denominator"]
                    choices.append(choice)
                    counts["eligible"] += 1
                    counts["selected"] += selected
                    if selected:
                        counts[carrier] += 1
                    samples += 1
                digest.update((json.dumps(dict(step=step, choices=choices), sort_keys=True, separators=(",", ":")) + "\n").encode())
        assert step == target_expected["batches"] == 10000
        assert counts["eligible"] == target_expected["eligible_slots"] == 74219
        assert counts["selected"] == target_expected["selected_slots"] == 37114
        assert digest.hexdigest() == target_expected["decision_identity_sha256"]
        results.append(dict(target=target, batches=step, eligible_slots=counts["eligible"],
                            selected_slots=counts["selected"], selected_per_carrier={c: counts[c] for c in ("light", "time", "timer")},
                            independent_integer_rng_draws=2 * samples,
                            decision_identity_sha256=digest.hexdigest(), exact_against_gpu=True))
    assert len(results) == 2 and {r["target"] for r in results} == set(TARGETS)
    assert results[0]["decision_identity_sha256"] == results[1]["decision_identity_sha256"]
    report = dict(
        schema="mac-common-carrier-prospective-decisions-independent-audit-v1",
        source_sha256=sha(Path(__file__).read_bytes()), gpu_source_commit=GPU_PIN,
        original_local_coordinate_audit_sha256=sha(prior_bytes),
        original_gpu_decision_result_sha256=sha(decision_bytes),
        original_gpu_native_execution_sha256=sha(native_bytes),
        original_gpu_policy_sha256=sha(policy), numpy_version=np.__version__,
        targets=results, model_invocations=0, audio_decodes=0,
        limits=[
            "Independent reconstruction of saved choices from original metadata/trace and PCG64; not a replacement tensor schedule.",
            "Native request identities checked, not native audio quality, waveform lengths, rendered features or their provenance.",
            "Integer midpoint crop mapping is prospective; actual new lengths, reachable support and tensor identity await new features.",
            "No training/model/new audio/fresh development/acceptance access; not evidence of recognition improvement.",
        ],
    )
    with output.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(report=str(output), targets=results)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repository-root", type=Path, required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--original-metadata", type=Path, required=True)
    ap.add_argument("--coordinate-audit", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    run(a.repository_root, a.packet, a.original_metadata, a.coordinate_audit, a.report)
