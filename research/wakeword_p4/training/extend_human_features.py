"""Create a fresh feature view with additional, prepartitioned human negatives.

Synthetic features, labels, PCM speed and existing partitions are preserved.
Extra training/validation/testing rows retain their original manifest splits.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--human-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh feature view")
    base = args.base_features.resolve()
    source = args.human_corpus.resolve()
    existing = json.loads((base/"human.json").read_text())
    manifest_path = source/"manifest.json"
    corpus = json.loads(manifest_path.read_text())
    records = corpus["records"]
    def identity(row):
        if "speaker" in row:
            return "speaker:"+str(row["speaker"])
        return "record:"+str(row["id"])

    existing_ids = {identity(r) for r in existing}
    new_ids = [identity(r) for r in records]
    if len(set(new_ids)) != len(new_ids) or existing_ids.intersection(new_ids):
        raise ValueError("Additional human source identities overlap existing partitions or one another")
    if corpus.get("speaker_identity_available") is False and any("speaker" in row for row in records):
        raise ValueError("Source does not provide individual speaker IDs")
    additions = []
    for row in records:
        if row["split"] not in ("training", "validation", "testing"):
            raise ValueError("New human source has no explicit training/validation/testing partition")
        path = Path(row["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Expected a source-relative feature path")
        if hashlib.sha256((source/path).read_bytes()).hexdigest() != row["feature_sha256"]:
            raise ValueError("Additional human feature checksum mismatch")
        additions.append({**row, "source_type": "speech", "path": str(Path("additional-human")/path)})
    args.output.mkdir(parents=True)
    for name in ("features.bin", "features.json", "dataset", "ambient", "ambient.json", "human-source"):
        path = base/name
        if not path.exists():
            raise ValueError(f"Missing original feature component: {name}")
        (args.output/name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (args.output/"additional-human").symlink_to(source, target_is_directory=True)
    # Anchor all previous rows to their previous feature view. Reusing the
    # additional-human name alone would redirect an earlier extension's rows
    # into the latest corpus, silently breaking chained feature views.
    (args.output/"base-human").symlink_to(base, target_is_directory=True)
    preserved = [{**row, "path": str(Path("base-human")/row["path"])} for row in existing]
    (args.output/"human.json").write_text(json.dumps(preserved+additions, indent=2)+"\n")
    report = {"scope": "Feature view only; original synthetic features and new human split assignments unchanged",
              "base_feature_index_sha256": hashlib.sha256((base/"features.json").read_bytes()).hexdigest(),
              "base_human_index_sha256": hashlib.sha256((base/"human.json").read_bytes()).hexdigest(),
              "additional_human_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "additional_speaker_identity_available": corpus.get("speaker_identity_available", True),
              "additional_hours": {split: sum(r["seconds"] for r in additions if r["split"] == split)/3600
                                   for split in ("training", "validation", "testing")}}
    (args.output/"generation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
