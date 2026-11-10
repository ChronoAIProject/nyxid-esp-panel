"""Append audited training-only human streams without changing evaluation rows."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--human-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, human, out = args.base.resolve(), args.human_features.resolve(), args.output.resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory; preserve partial runs")
    old = json.loads((base / "human.json").read_text())
    candidates = json.loads((human / "human.json").read_text())
    cache = json.loads((human / "cache.json").read_text())
    if sha(human / "human.json") != cache["index_sha256"]:
        raise ValueError("Human feature index differs from audited cache")
    additions = []
    speakers = {str(r["speaker"]) for r in old}
    for row in candidates:
        if row["split"] != "training":
            continue
        if str(row["speaker"]) in speakers:
            raise ValueError("Speaker already appears in base")
        if any(str(r["speaker"]) == str(row["speaker"]) and r["split"] != "training"
               for r in candidates):
            raise ValueError("Cross-split speaker")
        path = (human / row["path"]).resolve()
        if not path.is_relative_to(human):
            raise ValueError("Feature path escapes source")
        if sha(path) != row["feature_sha256"]:
            raise ValueError("Feature hash differs from audited source")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (row["frames"], 40) or array.dtype != np.dtype("uint16"):
            raise ValueError("Unexpected MicroFrontend array")
        if not row["first_two_chunk_continuity_checked"] or row["source_seconds"] <= 0:
            raise ValueError("Missing frontend continuity/exposure audit")
        additions.append({**row, "path": "human-additions/" + row["path"],
                          "category": "mandarin"})
        speakers.add(str(row["speaker"]))
    if not additions:
        raise ValueError("No training streams")
    out.mkdir(parents=True)
    for entry in base.iterdir():
        if entry.name not in ("human.json", "base-provenance.json"):
            (out / entry.name).symlink_to(entry.resolve())
    (out / "human-additions").symlink_to(human, target_is_directory=True)
    combined = old + additions
    (out / "human.json").write_text(json.dumps(combined, indent=2) + "\n")
    for split in ("validation", "testing"):
        assert [r for r in old if r["split"] == split] == [r for r in combined if r["split"] == split]
    report = {
        "role": "Training-only Mandarin ambient feature addition; not classifier evidence",
        "base_human_index_sha256": sha(base / "human.json"),
        "base_feature_index_sha256": sha(base / "features.json"),
        "base_feature_binary_sha256": sha(base / "features.bin"),
        "source_human_index_sha256": sha(human / "human.json"),
        "source_cache_sha256": sha(human / "cache.json"),
        "combined_human_index_sha256": sha(out / "human.json"),
        "source_sha256": sha(Path(__file__)),
        "added_streams": len(additions),
        "added_speakers": [r["speaker"] for r in additions],
        "added_original_hours": sum(r["source_seconds"] for r in additions) / 3600,
        "validation_and_development_regression_rows_unchanged": True,
        "keyword_feature_index_and_binary_unchanged": True,
        "independent_final_accessed": False,
        "source_evaluation_audio_accessed": False,
        "additions": additions,
    }
    (out / "base-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "additions"}, indent=2))


if __name__ == "__main__":
    main()
