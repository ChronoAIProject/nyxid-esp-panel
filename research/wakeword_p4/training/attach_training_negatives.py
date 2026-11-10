"""Attach audited training-only negatives; existing development fixtures stay unchanged."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pcm_hash(path):
    pcm, rate = sf.read(path, dtype="int16")
    return hashlib.sha256(str((rate, pcm.shape)).encode()+pcm.tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--negative-source", type=Path, action="append", required=True)
    p.add_argument("--backgrounds", type=Path, required=True,
                   help="Exclude source WAV/decoded PCM duplicates of existing environment fixtures")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    base = a.base.resolve()
    ambient = json.loads((base/"ambient.json").read_text())
    evaluation_before = [r for r in ambient if r["split"] != "training"]
    known_bytes = {r["source_sha256"] for r in ambient}
    known_pcm = set()
    for row in json.loads((a.backgrounds/"manifest.json").read_text())["records"]:
        path = a.backgrounds/row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError("Changed base environment audio")
        known_bytes.add(row["sha256"])
        known_pcm.add(pcm_hash(path))
    additions, excluded, sources = [], [], []
    counts, hours = Counter(), Counter()
    for number, source in enumerate(a.negative_source):
        source = source.resolve()
        report = json.loads((source/"preparation.json").read_text())
        if not report["all_training_only"] or sha(source/"ambient.json") != report["index_sha256"]:
            raise ValueError("Incomplete or changed prepared source")
        sources.append({"name": source.name, "preparation_sha256": sha(source/"preparation.json")})
        for row in json.loads((source/"ambient.json").read_text()):
            if row["split"] != "training" or row["window_step_ms"] != 10 or row["channels"] != 40:
                raise ValueError("Incompatible or non-training negative")
            path = source/row["path"]
            audio = source/row["audio_path"]
            if sha(path) != row["feature_sha256"] or sha(audio) != row["source_sha256"]:
                raise ValueError("Changed features or source audio")
            decoded = pcm_hash(audio)
            if row["source_sha256"] in known_bytes or decoded in known_pcm:
                excluded.append({"source_path": row["source_path"], "sha256": row["source_sha256"],
                                 "reason": "Exact source-byte or decoded-PCM duplicate"})
                continue
            data = np.load(path, mmap_mode="r")
            if data.dtype != np.uint16 or data.ndim != 2 or data.shape[1] != 40 or len(data) < 180:
                raise ValueError("Invalid frontend feature array")
            known_bytes.add(row["source_sha256"])
            known_pcm.add(decoded)
            additions.append({**row, "path": str(Path(f"negative-source-{number}")/row["path"]),
                              "audio_path": str(Path(f"negative-source-{number}")/row["audio_path"])})
            counts[row["category"]] += 1
            hours[row["category"]] += row["source_seconds"]/3600
    root = a.output.resolve()
    root.mkdir(parents=True)
    for name in ("features.bin", "features.json", "dataset", "human.json", "human-source", "ambient"):
        (root/name).symlink_to((base/name).resolve(), target_is_directory=(base/name).is_dir())
    for number, source in enumerate(a.negative_source):
        (root/f"negative-source-{number}").symlink_to(source.resolve(), target_is_directory=True)
    (root/"ambient.json").write_text(json.dumps(ambient+additions, indent=2)+"\n")
    assert [r for r in ambient+additions if r["split"] != "training"] == evaluation_before
    report = {"sources": sources, "added_counts": dict(counts), "added_source_hours": dict(hours),
              "duplicate_sources_excluded": excluded, "base_index_sha256": sha(base/"features.json"),
              "existing_feature_binary_reused": True, "existing_evaluation_records_unchanged": True,
              "base_ambient_index_sha256": sha(base/"ambient.json"), "all_additions_training_only": True,
              "source_sha256": sha(Path(__file__)),
              "limits": "Exact WAV/PCM deduplication does not establish independence of re-encoded, resampled or cropped source material. No new acceptance evidence."}
    (root/"negative-additions.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
