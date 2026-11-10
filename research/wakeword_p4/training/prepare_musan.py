"""Hash-pinned MUSAN parquet audio to training-only 10 ms MicroFrontend features."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from features import features


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, action="append", required=True,
                   help="Pinned source.json, beside its downloaded parquet files")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    sources, files = [], []
    for source_path in a.source:
        source = json.loads(source_path.read_text())
        if source["repo"] != "corypaik/musan" or source["revision"] != "76f9882cfa4475efe11508ac9aa32722f84ca5b7":
            raise ValueError("Unexpected mirror revision")
        sources.append(source)
        for row in source["files"]:
            path = source_path.parent/row["path"]
            if row["path"].split("/")[0] not in ("music", "noise"):
                raise ValueError("Speech excluded from this source plan")
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Changed parquet")
            files.append((row["path"], path))
    a.output.mkdir(parents=True)
    records, seen, skipped = [], set(), []
    counts, hours = Counter(), Counter()
    with (a.output/"audio.jsonl").open("w") as log:
        for relative, path in files:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for item in batch.to_pylist():
                    category = {0: "music", 2: "noise"}.get(item["label"])
                    if category != relative.split("/")[0] or not item["audio"]["bytes"]:
                        raise ValueError("Unexpected category or externally referenced audio")
                    data = item["audio"]["bytes"]
                    digest = hashlib.sha256(data).hexdigest()
                    if digest in seen:
                        skipped.append({"path": item["path"], "sha256": digest, "reason": "duplicate source bytes"})
                        continue
                    seen.add(digest)
                    pcm, rate = sf.read(io.BytesIO(data), dtype="int16")
                    if rate != 16000 or pcm.ndim != 1 or not len(pcm):
                        raise ValueError("Expected nonempty mono 16 kHz WAV")
                    source_seconds = len(pcm)/rate
                    # Short source clips are repeated to six seconds for usable
                    # training windows. Exposure always counts original audio only.
                    rendered = np.resize(pcm, max(len(pcm), 96000))
                    raw = features(rendered)
                    key = f"{category}-{len(records):05d}"
                    audio_path = Path("audio")/f"{key}.wav"
                    feature_path = Path("features")/f"{key}.npy"
                    (a.output/audio_path).parent.mkdir(exist_ok=True)
                    (a.output/feature_path).parent.mkdir(exist_ok=True)
                    (a.output/audio_path).write_bytes(data)
                    np.save(a.output/feature_path, raw)
                    row = {"split": "training", "path": str(feature_path),
                           "audio_path": str(audio_path), "source_path": item["path"],
                           "parquet": relative, "category": category, "source": item["source"],
                           "source_sha256": digest, "source_seconds": source_seconds,
                           "seconds": len(rendered)/rate, "feature_sha256": sha(a.output/feature_path),
                           "frames": len(raw), "window_step_ms": 10, "channels": 40}
                    records.append(row)
                    log.write(json.dumps(row)+"\n"); log.flush()
                    counts[category] += 1
                    hours[category] += source_seconds/3600
                    if len(records) % 100 == 0:
                        print(json.dumps({"prepared": len(records), "source_hours": dict(hours)}), flush=True)
    (a.output/"ambient.json").write_text(json.dumps(records, indent=2)+"\n")
    report = {"sources": sources, "counts": dict(counts), "source_hours": dict(hours),
              "duplicate_sources_skipped": skipped, "source_sha256": sha(Path(__file__)),
              "frontend_source_sha256": sha(Path(__file__).with_name("features.py")),
              "index_sha256": sha(a.output/"ambient.json"), "all_training_only": True,
              "limits": ["Music subset is the declared first two shards, not random genre-balanced sampling.",
                         "No speech category imported; vocal music and ambient speech can still occur and have not been exhaustively transcribed.",
                         "Mirror hashes establish the pinned mirror bytes, not an independent match against the entire official archive.",
                         "Per-source license/attribution files must accompany any redistributed audio; this report redistributes no audio."]}
    (a.output/"preparation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k,v in report.items() if k != "sources"}), flush=True)


if __name__ == "__main__":
    main()
