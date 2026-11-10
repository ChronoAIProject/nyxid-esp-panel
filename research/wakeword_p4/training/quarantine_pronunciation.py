"""Create a training-only pronunciation-screen ablation without relabeling.

Greedy phonemes are an imperfect screen, not pronunciation ground truth.
Retain every original audio/feature file and every held-out feature record.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical_prefix(phones):
    """The frozen rule: a Hey-Nicks phone sequence within the first 12 phones."""
    phones = phones[:12]
    for at in range(len(phones)-4):
        if (phones[at] in ("eɪ", "eː", "e", "ɛ")
                and phones[at+1] == "n"
                and phones[at+2] in ("ɪ", "i", "iː")
                and phones[at+3:at+5] == ["k", "s"]):
            return True
    return False


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def unique_paths(rows):
    result = {}
    for row in rows:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or str(path) in result:
            raise ValueError("Duplicate or unsafe source path")
        result[str(path)] = row
    return result


def create_view(base, audit, selection, output):
    if output.exists():
        raise ValueError("Use a fresh feature view")
    base = base.resolve()
    source = base/"dataset"
    manifest = source/"manifest.jsonl"
    originals = unique_paths(read_jsonl(manifest))
    selected = unique_paths(read_jsonl(selection))
    expected = {p for p, row in originals.items()
                if row["split"] == "training" and (row["label"] or row["text"].startswith("Hey "))}
    if set(selected) != expected:
        raise ValueError("Selection must cover exactly the prespecified training-only source rows")
    report = json.loads((audit/"report.json").read_text())
    if (report["source_manifest_sha256"] != digest(manifest)
            or report["selection_sha256"] != digest(selection) or report["split"] != "training"):
        raise ValueError("Audit is bound to another source/selection/split")
    predictions = unique_paths(read_jsonl(audit/"phonemes.jsonl"))
    if set(predictions) != expected or report["clips"] != len(expected):
        raise ValueError("Audit is incomplete or contains additional rows")
    quarantined = {}
    for path in sorted(expected):
        row, item, prediction = originals[path], selected[path], predictions[path]
        if (item["sha256"] != row["sha256"] or prediction["sha256"] != row["sha256"]
                or prediction["source_label"] != row["label"] or prediction["expected"] != row["text"]):
            raise ValueError("Audit and source audio/label/text identities differ")
        matches = canonical_prefix(prediction["phonemes"])
        if bool(row["label"]) != matches:
            quarantined[path] = {"path": path, "source_label": row["label"],
                                 "sha256": row["sha256"], "engine": row.get("engine", "piper"),
                                 "reason": "positive_without_pattern" if row["label"] else "ambiguous_negative_with_pattern",
                                 "phonemes": prediction["phonemes"]}
    records = json.loads((base/"features.json").read_text())
    # Bind all feature metadata to the unchanged manifest before excluding rows.
    for row in records:
        original = originals[row["path"]]
        if any(row[k] != original[k] for k in ("split", "label", "sha256", "text")):
            raise ValueError("Feature/source identity mismatch")
    retained = [r for r in records if not (r["split"] == "training" and r["path"] in quarantined)]
    engines_before = {r["engine"] for r in records if r["split"] == "training" and r["label"]}
    engines_after = {r["engine"] for r in retained if r["split"] == "training" and r["label"]}
    if not engines_before or engines_after != engines_before:
        raise ValueError("Pronunciation screen empties a positive training engine")
    heldout = [r for r in records if r["split"] != "training"]
    if heldout != [r for r in retained if r["split"] != "training"]:
        raise ValueError("Held-out feature records changed")
    human = json.loads((base/"human.json").read_text())
    preserved_human = [{**r, "path": str(Path("base-human")/r["path"])} for r in human]
    for row in human:
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts or not (base/path).is_file():
            raise ValueError("Unsafe or missing human feature path")
    for name in ("features.bin", "dataset", "ambient", "ambient.json"):
        if not (base/name).exists():
            raise ValueError(f"Missing base feature component: {name}")
    output.mkdir(parents=True)
    for name in ("features.bin", "dataset", "ambient", "ambient.json"):
        target = base/name
        (output/name).symlink_to(target.resolve(), target_is_directory=target.is_dir())
    if (base/"human-source").exists():
        (output/"human-source").symlink_to((base/"human-source").resolve(), target_is_directory=True)
    (output/"base-human").symlink_to(base, target_is_directory=True)
    (output/"features.json").write_text(json.dumps(retained)+"\n")
    (output/"human.json").write_text(json.dumps(preserved_human, indent=2)+"\n")
    # Verify chained feature views still resolve each human/environment target.
    if any((output/new["path"]).resolve() != (base/old["path"]).resolve()
           for old, new in zip(human, preserved_human)):
        raise ValueError("Human feature targets changed")
    (output/"quarantine.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n"
                                               for r in quarantined.values()))
    def counts(rows):
        return dict(sorted(Counter(f"{r['split']}/{r['label']}/{r['engine']}" for r in rows).items()))
    result = {"scope": "Training-only experimental quarantine; no source label, PCM, feature byte or held-out row changed",
              "rule": "Within first 12 greedy phones: optional h, eɪ/eː/e/ɛ, n, ɪ/i/iː, k, s",
              "limits": "Recognizer predictions are imperfect; this screen is not phonetic ground truth or model acceptance",
              "base_feature_index_sha256": digest(base/"features.json"),
              "source_manifest_sha256": digest(manifest), "selection_sha256": digest(selection),
              "audit_report_sha256": digest(audit/"report.json"),
              "audit_predictions_sha256": digest(audit/"phonemes.jsonl"),
              "recognizer": report["recognizer"], "quarantined_sources": len(quarantined),
              "quarantine_by_reason": dict(Counter(r["reason"] for r in quarantined.values())),
              "before": counts(records), "after": counts(retained),
              "heldout_records": len(heldout), "heldout_records_sha256": json_digest(heldout),
              "preservation": {"heldout_records_exact": True, "feature_binary_same_target": True,
                               "source_audio_same_target": True, "human_targets_exact": True,
                               "ambient_index_and_directory_same_targets": True}}
    (output/"generation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_view(args.base_features, args.audit, args.selection, args.output),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
