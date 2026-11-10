"""Check stream feature lengths, provenance and source identity partitions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.features
    records = json.loads((root/"features.json").read_text())
    voices, sources, variants = {}, {}, set()
    counts = Counter()
    offset = 0
    for row in records:
        if row["offset"] != offset or row["frames"] < 180:
            raise ValueError(f"Non-contiguous/short sequence: {row['path']}")
        offset += row["frames"]
        if row["split"] not in ("training", "validation", "testing"):
            raise ValueError("Unknown split")
        key = (row["path"], row["variant"])
        if key in variants:
            raise ValueError(f"Duplicate rendering: {key}")
        variants.add(key)
        identities = [(row["engine"], str(v)) for v in row["speakers"]]
        for identity in identities:
            if identity in voices and voices[identity] != row["split"]:
                raise ValueError(f"Voice crosses splits: {identity}")
            voices[identity] = row["split"]
        source = row["sha256"]
        if source in sources and sources[source] != row["split"]:
            raise ValueError("Source waveform crosses splits")
        sources[source] = row["split"]
        if row["label"] and not 0 <= row["keyword_frame"] < row["frames"]:
            raise ValueError("Keyword outside sequence")
        counts[f"{row['split']}/{row['engine']}/{row['label']}"] += 1
    if (root/"features.bin").stat().st_size != offset*40*2:
        raise ValueError("Truncated or oversized binary features")
    exposure, human_ids = Counter(), {}
    for filename in ("human.json", "ambient.json"):
        for row in json.loads((root/filename).read_text()):
            x = np.load(root/row["path"], mmap_mode="r")
            if x.ndim != 2 or x.shape[1] != 40 or len(x) < 180 or x.dtype != np.uint16:
                raise ValueError(f"Unexpected negative features: {row['path']}")
            identity = (filename, row.get("speaker", row.get("source_sha256")))
            if identity[1] is None:
                raise ValueError("Missing negative source identity")
            if identity in human_ids and human_ids[identity] != row["split"]:
                raise ValueError("Negative speaker/blob crosses splits")
            human_ids[identity] = row["split"]
            exposure[f"{row['split']}/{filename}"] += row.get("source_seconds", row["seconds"])/3600
    report = {"sequences": len(records), "frames": offset, "source_waveforms": len(sources),
              "counts": dict(counts), "negative_feature_hours": dict(exposure),
              "voice_and_source_split_checks_passed": True,
              "features_sha256": sha256(root/"features.bin"),
              "index_sha256": sha256(root/"features.json"),
              "limits": "Variants are correlated, not independent trials. Voice IDs do not establish cross-engine physical identity. Source hours exclude inserted utterance-boundary silence and frontend padding."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
