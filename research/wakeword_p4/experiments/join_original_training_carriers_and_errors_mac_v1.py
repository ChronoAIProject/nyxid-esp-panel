"""Join authenticated original training carrier mass to consumed false clips."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


BASE = Path("research/wakeword_p4/results")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def literal_key(text):
    # Match textual case/final punctuation only, not pronunciation or synonyms.
    return text.casefold().strip().rstrip(".,!? ")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "development-packet", "census", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    repo = a.repository_root.resolve()
    census_bytes = a.census.read_bytes()
    gpu_census = subprocess.check_output(["git", "show", "7da71cda:research/wakeword_p4/results/4090-hey-broad-metadata-sampling-mass-v1.json"], cwd=repo)
    assert census_bytes == gpu_census
    census = json.loads(census_bytes)
    errors_path = repo / BASE / "mac-fp32-reused-all-point-persistent-errors-v1.json"
    errors_bytes = errors_path.read_bytes()
    errors = json.loads(errors_bytes)
    intake_bytes = (repo / BASE / "mac-fp32-reused-diagnostic-intake-v1.json").read_bytes()
    assert sha(intake_bytes) == errors["intake_report_sha256"]
    intake = json.loads(intake_bytes)
    assert intake["all_passed"]
    manifest_bytes = (a.development_packet / "manifest.json").read_bytes()
    assert sha(manifest_bytes) == intake["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    data = (a.development_packet / "materials/synthetic.json").read_bytes()
    assert sha(data) == manifest["files"]["materials/synthetic.json"]["sha256"] == errors["synthetic_manifest_sha256"]
    rows = json.loads(data)["records"]
    by_id = {(r["source_id"], r["variant"]): r for r in rows}
    assert len(rows) == len(by_id) == 360
    common = [r for r in rows if r["target_id"] not in ("hey_willow", "hey_nyx")]
    assert len(common) == 288
    primary = next(m for m in errors["models"] if m["model"] == "hey_nyx-float_control")
    assert primary["registered_points"] == 138 and primary["false_clips_at_every_registered_point"] == 13
    persistent = primary["persistent_common_false_records"]
    for row in persistent:
        source = by_id[row["source_id"], row["variant"]]
        assert all(source[k] == v for k, v in row.items())
    train = census["results"]["hey_nyx"]
    assert train["row_roles"] == dict(own=25128, reciprocal=25128, common=56700)
    common_training = [r for r in train["details"] if r["slot"] == "ordinary_negative_surviving" and r["source_role"] == "common"]
    assert sum(r["rows"] for r in common_training) == 56700
    assert {r["carrier"] for r in common_training} == {"bare"}
    reciprocal_commands = [r for r in train["details"] if r["slot"] == "ordinary_negative_surviving"
                           and r["source_role"] == "reciprocal" and r["carrier"] != "bare"]
    assert reciprocal_commands
    details = []
    for row in persistent:
        matching = [r for r in common_training if literal_key(r["text"]) == literal_key(row["literal"])]
        assert matching
        details.append(dict(source_id=row["source_id"], variant=row["variant"], literal=row["literal"],
                            development_carrier=row["carrier"] or "bare",
                            training_common_bare_rows=sum(r["rows"] for r in matching),
                            training_common_nonbare_rows=0,
                            expected_common_phrase_slots=str(sum((Fraction(r["exact_slots"]) for r in matching), Fraction()))))
    dev_carriers = Counter(r["carrier"] or "bare" for r in common)
    false_carriers = Counter(r["carrier"] or "bare" for r in persistent)
    result = dict(schema="mac-original-training-carrier-and-persistent-error-join-v1",
                  source_sha256=sha(Path(__file__).read_bytes()), census_sha256=sha(census_bytes),
                  original_metadata_inputs=census["inputs"], errors_sha256=sha(errors_bytes),
                  synthetic_manifest_sha256=sha(data), primary_model=primary["model"],
                  common_training_rows=56700, common_training_nonbare_rows=0,
                  reciprocal_command_negatives_present=True,
                  reused_common_development_clips=288, persistent_false_clips=13,
                  persistent_nonbare_false_clips=sum(bool(r["carrier"]) for r in persistent),
                  persistent_bare_false_clips=sum(not r["carrier"] for r in persistent),
                  by_development_carrier={k: dict(clips=dev_carriers[k], persistent_false_clips=false_carriers[k]) for k in sorted(dev_carriers)},
                  details=details, model_invocations=0, audio_decodes=0,
                  thresholds_selected=False, acceptance_scored=False,
                  limits="Actual original metadata and already-consumed diagnostic join, not observed window content or causal proof. All development clips share one Semaine TTS voice; variants are correlated. Eleven persistent errors have command carriers, but two bare errors remain. Missing common carrier coverage cannot alone explain all errors. Reciprocal command and own-prefix negatives are distinct existing supervision and are not erased by this report. No new data/training/model/boundary/threshold changes.")
    assert result["persistent_nonbare_false_clips"] == 11 and result["persistent_bare_false_clips"] == 2
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("details", "original_metadata_inputs")}, indent=2))


if __name__ == "__main__":
    main()
