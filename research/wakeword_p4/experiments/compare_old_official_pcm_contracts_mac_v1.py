"""Join original official/custom input contracts; JSON only, no scoring/media."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


REF = "f312c29"
R = "research/wakeword_p4/results/"
SYNTHETIC = ".local/wakeword/hey-development-replay-v1.json"
BACKGROUND = ".local/wakeword/hey-paired-earnings-development-v1/manifest.json"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    check(not args.report.exists(), "Do not overwrite an original result")
    ref = subprocess.check_output(["git", "-C", str(args.repository_root), "rev-parse", REF], text=True).strip()
    hashes = {}

    def read(name):
        raw = subprocess.check_output(["git", "-C", str(args.repository_root), "show", f"{ref}:{R}{name}"])
        hashes[R + name] = digest(raw)
        return json.loads(raw)

    official = read("4090-hey-official-development-execution-v1.json")
    old = read("4090-old-fixed-grid-regression-plan-v2.json")
    custom = read("4090-hey-development-custom-execution-v1.json")
    collection = read("4090-old-fixed-grid-regression-collection-v2.json")
    background_contract = read("4090-hey-official-background-inputs-v1.json")
    inventory = read("4090-hey-same-recipe-evidence-inventory-v1.json")
    for name, expected in inventory["files"].items():
        raw = subprocess.check_output(["git", "-C", str(args.repository_root), "show", f"{ref}:{name}"])
        check(digest(raw) == expected, "Original model inventory source binding")
    check(collection["plan_sha256"] == hashes[R + "4090-old-fixed-grid-regression-plan-v2.json"],
          "Old saved collection uses original plan")
    for path in (SYNTHETIC, BACKGROUND):
        check(official["files"][path] == old["files"][path] == custom["files"][path],
              "Same manifest identity in all three original executions")
    raw = args.synthetic_manifest.read_bytes()
    check(digest(raw) == official["files"][SYNTHETIC], "Actual synthetic manifest original bytes")
    synthetic = json.loads(raw)["records"]
    check(len(synthetic) == 360, "Original360, not future396 development")
    synthetic_official = [row for row in official["records"] if row["phase"] == "synthetic"]
    check(synthetic_official == [dict(row, phase="synthetic") for row in synthetic],
          "All original360 records/PCM hashes/order/boundaries/roles exactly equal")
    background = [row for row in official["records"] if row["phase"] == "background"]
    contracts = {row["source_id"]: row for row in background_contract["records"]}
    check(len(background) == len(contracts) == 12 and
          {row["source_id"] for row in background} == set(contracts), "Exact original12 background sources")
    check(background_contract["manifest_sha256"] == official["files"][BACKGROUND],
          "Original decoded-PCM contract belongs to same background manifest")
    total_samples = 0
    for row in official["records"]:
        check(old["files"][row["path"]] == row["sha256"] == official["files"][row["path"]],
              "Same original encoded WAV identity in both executions")
        if row["phase"] == "background":
            original = contracts[row["source_id"]]
            check((row["samples"], row["pcm_sha256"]) == (original["samples"], original["pcm_sha256"]),
                  "Same complete decoded background PCM/sample count")
            check(old["files"][row["frontend_path"]] == row["frontend_sha256"],
                  "Old background feature identity is execution-locked")
            total_samples += row["samples"]
    check(total_samples == background_contract["total_samples"] == 603875222, "Full background exposure")
    hours = total_samples / 16000 / 3600
    check(hours == official["background_hours"], "Same original exposure denominator")
    models = {model["id"] for model in old["models"]}
    check(len(models) == 5 and len(collection["records"]) == 1860, "Five complete original collections")
    expected = {(model, phase, index) for model in models for phase, rows in
                (("synthetic", synthetic_official), ("background", background)) for index in range(len(rows))}
    seen, feature_identities = set(), {}
    for record in collection["records"]:
        key = (record["model"], record["phase"], record["index"])
        check(key in expected and key not in seen, "Exactly one original collection record per input/model")
        seen.add(key)
        row = (synthetic_official if record["phase"] == "synthetic" else background)[record["index"]]
        frames = (row["samples"] + 159) // 160 - 2
        check(record["source_id"] == row["source_id"] and record["samples"] == row["samples"] and
              record["raw_frames"] == frames and record["final_feature_padding"] == (-frames) % 3 and
              record["steps"] == (frames + 2) // 3 and record["repeat_exact"], "Complete original input accounting")
        input_key = (record["phase"], record["index"])
        check(feature_identities.setdefault(input_key, record["raw_sha256"]) == record["raw_sha256"],
              "Same recorded raw feature identity across all five old models")
    check(seen == expected, "No missing original input/model")
    check(official["streams"] == len(official["records"]) * len(official["models"]) *
          len(official["threshold_requests"]) == 9672, "Original2x13x372 official grid")
    result = dict(schema="mac-old-official-pcm-contract-comparison-v1", source_commit=ref,
                  reader_sha256=digest(Path(__file__).read_bytes()), source_hashes=hashes,
                  inventory_source_hashes_checked=len(inventory["files"]),
                  common_manifests={p: official["files"][p] for p in (SYNTHETIC, BACKGROUND)},
                  synthetic_records_exact=360, synthetic_target_counts=dict(Counter(r["target_id"] for r in synthetic)),
                  background_contracts_exact=12, encoded_wav_identities_exact=372,
                  background_samples=total_samples, background_hours=hours,
                  original_collection_records_complete=1860, original_unique_feature_identities=372,
                  official_models=official["models"], official_original_streams=9672,
                  official_threshold_requests=official["threshold_requests"],
                  shared_pcm_contracts=True, model_invocations=0, media_reads=0,
                  limits="Original already-consumed360+12 declared PCM contracts/source roles/order/metadata only. No new decode or original model calculation reproduced. No official calibration/result/return stream audited here. Future396/Jenny/Muse is separate. Does not prove recall, FAR, latency, P4 or independent acceptance parity.")
    with args.report.open("x") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("synthetic_records_exact", "background_contracts_exact",
                     "background_hours", "original_collection_records_complete", "shared_pcm_contracts")}, indent=2))


if __name__ == "__main__":
    main()
