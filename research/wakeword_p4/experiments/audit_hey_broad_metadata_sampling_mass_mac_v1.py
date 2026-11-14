"""Census authenticated original training metadata; no feature/model/media reads.

Expected sampler mass is distinct from observed draws. Ordinary negative slots
32..47 are discarded by prefix replacement and are excluded from effective mass.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


PIN = "e1f341a9e2dc8dd073efd1f50e79d4912edbfed5"
BASE = "research/wakeword_p4"
TRAINER_SHA = "2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904"
POLICY_SHAS = {
    "experiments/audit_hey_broad_batches_v1.py": "a2c85e558f786961d6a3b5889390269394a91d2bfc9dd84522eecf409765442e",
    "experiments/schedule_hey_broad_v1.py": "e6df66f95ed41943cc7a6a5aee7c98cec352dbbf9e354dbce9d2362ec0c22f10",
    "experiments/run_hey_broad_paired_qat_v1.py": "ceab0f8828d2934c887e0e222bca1d955abec39cf93df18c7532c036bb8c7147",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, path):
    return subprocess.check_output(["git", "show", f"{PIN}:{BASE}/{path}"], cwd=repo)


def role(row, target):
    if row["target_id"] == target:
        return "own"
    return "common" if row["target_id"] == "common_negative" else "reciprocal"


def reduce_rows(rows, target):
    groups = defaultdict(list)
    for row in rows:
        assert row["split"] == "training"
        assert row["label"] == int(row["target_id"] == target)
        groups[row["label"], row["engine"]].append(row)
    engines = {label: sorted(e for lab, e in groups if lab == label) for label in (0, 1)}
    assert all(engines.values())
    prefix_engines = [e for e in engines[1] if e in ("piper", "kokoro")]
    assert prefix_engines
    output = []
    masses = Counter()
    # Source exponent zero gives each eligible engine equal probability; each
    # row within an engine is uniform. Surviving ordinary negatives number 32.
    for slot, label, eligible, slots, weight in (
        ("positive", 1, engines[1], 32, 1),
        ("ordinary_negative_surviving", 0, engines[0], 32, 20),
        ("prefix_negative", 1, prefix_engines, 16, 20),
    ):
        for engine in eligible:
            source = groups[label, engine]
            buckets = defaultdict(list)
            for row in source:
                pair = row["source_pair_key"]
                assert pair["engine"] == engine
                key = role(row, target), str(pair["carrier"]), row["text"]
                buckets[key].append(row)
            for (source_role, carrier, text), selected in sorted(buckets.items()):
                mass = Fraction(slots * len(selected), len(eligible) * len(source))
                masses[slot] += mass
                output.append(dict(
                    slot=slot, label=1 if slot == "positive" else 0,
                    source_role=source_role, engine=engine, carrier=carrier, text=text,
                    rows=len(selected), source_ids=len({r["source_id"] for r in selected}),
                    conditioning_ids=len({str(r["source_pair_key"]["conditioning"]) for r in selected}),
                    expected_slots_per_batch=float(mass), exact_slots=str(mass),
                    expected_weight_mass=float(weight * mass),
                ))
    assert masses == Counter(positive=32, ordinary_negative_surviving=32, prefix_negative=16)
    summaries = defaultdict(Fraction)
    for item in output:
        key = item["slot"], item["source_role"], item["carrier"]
        summaries[key] += Fraction(item["exact_slots"])
    return dict(
        rows=len(rows), row_roles=dict(Counter(role(r, target) for r in rows)),
        rows_by_label_and_engine=[dict(label=lab, engine=eng, rows=len(rs))
                                  for (lab, eng), rs in sorted(groups.items())],
        expected_effective_slots={k: str(v) for k, v in sorted(masses.items())},
        expected_slot_role_carrier=[dict(slot=k[0], source_role=k[1], carrier=k[2],
                                        exact_slots=str(v), expected_slots_per_batch=float(v))
                                    for k, v in sorted(summaries.items())],
        details=output,
    )


def self_test():
    def row(engine, target_id, index, carrier="bare"):
        return dict(split="training", label=int(target_id == "hey_nyx"),
                    engine=engine, target_id=target_id, source_id=f"{engine}-{index}",
                    text=target_id, source_pair_key=dict(engine=engine, carrier=carrier,
                                                        conditioning="constructed"))
    # 100:1 corpus imbalance still yields 1:1 negative engine mass. There are
    # 48 ordinary draws, but only 32 survive; prefix rows must carry label zero.
    rows = [row("piper", "common_negative", i) for i in range(100)]
    rows += [row("kokoro", "hey_willow", 100)]
    rows += [row("piper", "hey_nyx", 101), row("kokoro", "hey_nyx", 102, "command")]
    result = reduce_rows(rows, "hey_nyx")
    neg = [d for d in result["details"] if d["slot"] == "ordinary_negative_surviving"]
    assert {d["source_role"]: d["exact_slots"] for d in neg} == {"common": "16", "reciprocal": "16"}
    prefix = [d for d in result["details"] if d["slot"] == "prefix_negative"]
    assert len(prefix) == 2 and all(d["label"] == 0 and d["exact_slots"] == "8" for d in prefix)
    assert sum(Fraction(d["exact_slots"]) for d in result["details"]) == 80
    # Scaling one engine's corpus preserves its sampling mass.
    scaled = reduce_rows(rows[:100] * 3 + rows[100:], "hey_nyx")
    assert scaled["expected_slot_role_carrier"] == result["expected_slot_role_carrier"]
    print("constructed engine-weighting, replacement and label checks passed; no corpus census executed")


def run(repo, root, report):
    assert not report.exists()
    trainer = git_bytes(repo, "training/train_stream.py")
    assert digest(trainer) == TRAINER_SHA
    for path, expected in POLICY_SHAS.items():
        assert digest(git_bytes(repo, path)) == expected
    feature_report_bytes = git_bytes(repo, "results/4090-hey-broad-features-v1.json")
    feature_report = json.loads(feature_report_bytes)
    results = {}
    receipts = {}
    for target in ("hey_willow", "hey_nyx"):
        provenance_bytes = (root / target / "target-provenance.json").read_bytes()
        provenance = json.loads(provenance_bytes)
        assert provenance == feature_report["targets"][target]
        metadata = (root / target / "features.json").read_bytes()
        assert digest(metadata) == provenance["files"]["features.json"]
        rows = json.loads(metadata)
        assert len(rows) == 106956 and sum(r["label"] for r in rows) == 25128
        assert all(r["frames"] >= 225 for r in rows)
        results[target] = reduce_rows(rows, target)
        assert results[target]["row_roles"] == dict(own=25128, reciprocal=25128, common=56700)
        receipts[target] = dict(metadata_sha256=digest(metadata), provenance_sha256=digest(provenance_bytes))
        del rows, metadata
    result = dict(
        schema="mac-hey-broad-original-metadata-sampling-mass-v1", source_commit=PIN,
        source_sha256=digest(Path(__file__).read_bytes()), trainer_sha256=TRAINER_SHA,
        policy_source_sha256=POLICY_SHAS,
        original_feature_report_sha256=digest(feature_report_bytes), inputs=receipts,
        results=results, discarded_ordinary_negative_slots_per_batch=16,
        ambient_slots_per_batch=48, ambient_source_census_performed=False,
        expected_batch_labels=dict(positive=32, negative=96),
        expected_batch_weight_mass=dict(positive=32, negative=1920),
        limits=[
            "Authenticated metadata and exact source-selection expectations, not observed training draws or window endpoints.",
            "Conditioning/source IDs and rendered variants are not independent human voices.",
            "Does not verify pronunciations, crop content, actual audio/features, gradients, recognition or causality.",
            "No feature binary, ambient arrays, audio, model, heldout development or acceptance data read.",
        ],
    )
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({t: {k: v for k, v in r.items() if k != "details"} for t, r in results.items()}))


def published_derivation(repo, report):
    """Derive engine mass from published counts; do not claim a private-pack census."""
    assert not report.exists()
    sources = {}

    def read(name):
        path = "results/" + name
        data = git_bytes(repo, path)
        sources[path] = digest(data)
        return json.loads(data)

    for path, expected in POLICY_SHAS.items():
        assert digest(git_bytes(repo, path)) == expected
        sources[path] = expected
    assert digest(git_bytes(repo, "training/train_stream.py")) == TRAINER_SHA
    sources["training/train_stream.py"] = TRAINER_SHA
    reports = {e: read(f"4090-hey-broad-{e}-generation-v1.json")
               for e in ("piper", "kokoro", "qwen")}
    for r in reports.values():
        assert r["quality_pass"] == r["cases"] and not r["failures"]
    qwen = read("4090-hey-broad-qwen-standalone-material-plan-v1.json")
    qcounts = Counter(r["target"] if r["role"] == "own-target-positive" else "common_negative"
                      for r in qwen["requests"])
    assert sum(qcounts.values()) == reports["qwen"]["cases"] == 360
    assert qcounts["hey_willow"] == qcounts["hey_nyx"] == qwen["positive_per_arm"] == 72
    assert qcounts["common_negative"] == qwen["common_negatives"] == 216
    kokoro = read("4090-hey-broad-kokoro-generation-plan-v1.json")
    kcounts = Counter(r["request"]["target"] for r in kokoro["requests"])
    assert sum(kcounts.values()) == reports["kokoro"]["cases"] == 480
    assert kcounts["hey_willow"] == kcounts["hey_nyx"] == 240
    feature = read("4090-hey-broad-features-v1.json")
    render = read("4090-hey-broad-render-plan-v1.json")
    assert len(render["conditions"]) == 3
    # The renderer requires paired source keys for all positive sources. The
    # private active metadata still requires the separate run() census.
    own = {"piper": reports["piper"]["cases"] // 2, "kokoro": kcounts["hey_nyx"], "qwen": qcounts["hey_nyx"]}
    assert reports["piper"]["cases"] % 2 == 0
    assert sum(own.values()) * 3 == feature["targets"]["hey_nyx"]["positive_rows"] == 25128
    common = {"qwen": qcounts["common_negative"]}
    for engine in ("piper", "kokoro"):
        r = read(f"4090-hey-broad-common-{engine}-generation-v1.json")
        assert r["quality_pass"] == r["cases"] and not r["failures"]
        common[engine] = r["cases"]
    assert sum(common.values()) * 3 == 56700
    assert (sum(own.values()) + sum(common.values())) * 3 == feature["targets"]["hey_nyx"]["negative_rows"] == 81828
    result = dict(
        schema="mac-hey-broad-published-count-engine-mass-derivation-v1", source_commit=PIN,
        source_sha256=digest(Path(__file__).read_bytes()), input_source_sha256=sources,
        own_source_containers_per_word=own, common_negative_source_containers=common,
        rendering_variants_per_source=3,
        main_positive_slots_per_engine=str(Fraction(32, 3)),
        per_container_qwen_to_piper_sampling_ratio=str(Fraction(own["piper"], own["qwen"])),
        per_container_kokoro_to_piper_sampling_ratio=str(Fraction(own["piper"], own["kokoro"])),
        expected_ordinary_negative_common_slots=str(sum((Fraction(32, 3) * Fraction(common[e], common[e] + own[e]) for e in own), Fraction())),
        expected_ordinary_negative_reciprocal_slots=str(sum((Fraction(32, 3) * Fraction(own[e], common[e] + own[e]) for e in own), Fraction())),
        scope="Published completed-count/source-policy derivation; not an independently read active feature-pack census or observed training draw trace.",
        limits=["Source containers, rendered variants and conditioning IDs are not independent human voices.",
                "Equal engine weighting is an explicit policy, not itself a bug or causal proof of recognition failures.",
                "No arrays, models, audio, independent development or acceptance data read; no sampling/training policy changed."],
    )
    assert Fraction(result["expected_ordinary_negative_common_slots"]) + Fraction(result["expected_ordinary_negative_reciprocal_slots"]) == 32
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "input_source_sha256"}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--published-count-derivation", action="store_true")
    p.add_argument("--repository-root", type=Path)
    p.add_argument("--training-feature-root", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.self_test:
        self_test()
    elif a.published_count_derivation:
        if not all((a.repository_root, a.report)):
            p.error("repository-root and report are required")
        published_derivation(a.repository_root, a.report)
    else:
        if not all((a.repository_root, a.training_feature_root, a.report)):
            p.error("repository-root, training-feature-root and report are required")
        run(a.repository_root, a.training_feature_root, a.report)


if __name__ == "__main__":
    main()
