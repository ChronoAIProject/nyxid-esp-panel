"""Check training-only coverage and pilot metadata against original Mac records.

This does not load source audio, features, models, or unused evaluation data.
The GPU pilot waveform-quality claims require its original arrays separately.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


REPO = Path(__file__).resolve().parents[2]
GPU_COMMIT = "636fec1"
PREFIX = "research/wakeword_p4/"
SOURCE_SHA = "0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981"
RECORDS_SHA = "757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output path required")
    require(np.__version__ == "2.4.6", "Unreviewed NumPy partition implementation")
    blobs = {}

    def blob(relative):
        data = subprocess.check_output(
            ["git", "-C", str(REPO), "show", GPU_COMMIT + ":" + PREFIX + relative]
        )
        blobs[relative] = sha(data)
        return data

    original = args.original_metadata
    source_bytes = (original / "sources.json").read_bytes()
    require(sha(source_bytes) == SOURCE_SHA, "Original source metadata differs")
    require(sha((original / "records.json").read_bytes()) == RECORDS_SHA,
            "Original training-record metadata differs")
    sources = json.loads(source_bytes)
    coverage = json.loads(blob("results/4090-piper-training-phrase-coverage-v1.json"))
    ids = np.random.default_rng(726).permutation(904)[:672].tolist()
    require(ids == coverage["training_ids"] and ids[:32] == coverage["pilot_ids"],
            "Original partition differs")
    allowed = set(ids)
    summaries = {}
    for phrase in ("hey nyx", "hey nick", "hey next", "hey mix"):
        # Enumerate every alias rather than assuming the first alias is complete.
        selected = []
        for index, source in enumerate(sources):
            aliases = [a for a in source["aliases"]
                       if a["engine"] == "piper" and a["phrase"] == phrase]
            if aliases:
                require(len(aliases) == len(source["aliases"]) == 1,
                        "Piper alias assumption requires a new review")
                selected.append((index, aliases[0]))
        counts = Counter(int(sid) for _, a in selected for sid in set(a["speakers"]))
        require(set(counts) <= allowed, "Non-training conditioning ID")
        actual = dict(
            canonical_sources=len(selected), conditioning_ids=len(counts),
            missing_training_ids=sorted(allowed - set(counts)),
            source_indices=[index for index, _ in selected],
            counts={str(sid): counts[sid] for sid in ids},
        )
        require(actual == coverage["groups"][phrase], "Coverage ledger differs")
        summaries[phrase] = {key: actual[key] for key in
                             ("canonical_sources", "conditioning_ids")}

    run_bytes = blob("results/4090-piper-balanced-training-pilot-run-v1.json")
    manifest_bytes = blob("results/4090-piper-balanced-training-pilot-manifest-v1.jsonl")
    report = json.loads(blob("results/4090-piper-balanced-training-pilot-original-v1.json"))
    run = json.loads(run_bytes)
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    require(sha(run_bytes) == report["run_sha256"]
            and sha(manifest_bytes) == report["manifest_sha256"], "Pilot receipt differs")
    require(sha(blob("experiments/generate_piper_balanced_training_pilot_v1.py"))
            == report["script_sha256"] == run["script_sha256"], "Generator differs")
    require(sha(blob("experiments/piper_balanced_training_pilot_v1.md"))
            == run["plan_sha256"], "Pilot plan differs")
    require(blobs["results/4090-piper-training-phrase-coverage-v1.json"]
            == run["coverage_sha256"], "Coverage receipt differs")
    phrases = [("hey_nyx", "Hey Nicks."), ("hey_nick", "Hey Nick."),
               ("hey_next", "Hey Next."), ("hey_mix", "Hey Mix.")]
    grid = [(sid, speed, target, text) for sid in ids[:32]
            for speed in (.9, 1., 1.1) for target, text in phrases]
    require(len(rows) == len(grid) == report["clips"] == 384, "Pilot count differs")
    require(run["ids"] == ids[:32] and run["seed"] == 187726,
            "Pilot IDs or seed differs")
    phones = {}
    for order, (row, expected) in enumerate(zip(rows, grid)):
        require((row["speaker_id"], row["length_scale"], row["target"], row["text"])
                == expected and row["order"] == order, "Pilot grid differs")
        require(row["paired_speaker_ids"] == [expected[0]] * 2
                and row["split"] == "training" and row["draw"] == 0
                and not row["trimmed"] and not row["normalized"]
                and row["run_sha256"] == report["run_sha256"], "Pilot identity differs")
        require(sum(row["duration_samples"]) == row["native_samples"]
                and len(row["duration_samples"]) == len(row["token_ids"])
                and row["samples"] == (row["native_samples"] * 320 + 440) // 441,
                "Recorded duration or resample ledger differs")
        require(row["quality_pass"] == all(row["checks"].values()),
                "Recorded quality flags differ")
        phones.setdefault(row["target"], set()).add(tuple(row["phonemes"]))
    require(all(len(p) == 1 for p in phones.values())
            and len({next(iter(p)) for p in phones.values()}) == 4,
            "Recorded phoneme sequences do not distinguish the four phrases")
    require(report["quality_pass"] == sum(r["quality_pass"] for r in rows) == 384
            and report["failures"] == [], "Pilot recorded failures differ")
    result = dict(
        scope="Training coverage and 384-pilot metadata replay only",
        all_metadata_checks_pass=True, gpu_commit=GPU_COMMIT,
        original_sources_sha256=SOURCE_SHA, original_records_sha256=RECORDS_SHA,
        training_conditioning_ids=len(ids), all_alias_coverage_exact=summaries,
        pilot_grid_rows_exact=len(rows),
        pilot_phrase_counts=dict(Counter(r["target"] for r in rows)),
        pilot_recorded_phones={t: list(next(iter(p))) for t, p in phones.items()},
        pilot_duration_ledgers_exact=len(rows),
        original_gpu_blobs_sha256=blobs,
        script_sha256=sha(Path(__file__).read_bytes()), numpy=np.__version__,
        limits=["No waveform, raw feature, model, score, microphone or speaker access.",
                "Pilot file hashes, conversion and mechanical quality not replayed on Mac.",
                "Conditioning-ID membership is not independent-person or acoustic coverage.",
                "Recorded phonemes and durations do not establish acoustic pronunciation.",
                "No recognition improvement, causal explanation or P4 eligibility established."],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("original_gpu_blobs_sha256", "pilot_recorded_phones")}, indent=2))


if __name__ == "__main__":
    main()
