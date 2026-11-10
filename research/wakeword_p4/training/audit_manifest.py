"""Verify waveform provenance, labels and whole-voice split isolation."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import soundfile as sf


def audit(root):
    if not (root / "generation.json").is_file():
        raise ValueError("Generation is incomplete")
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("Empty dataset")
    voices, hashes, paths = {}, {}, set()
    counts, seconds = Counter(), Counter()
    for row in rows:
        split = row["split"]
        if split not in ("training", "validation", "testing") or row["label"] not in (0, 1):
            raise ValueError(f"Invalid split/label: {row}")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or path in paths:
            raise ValueError(f"Escaping or duplicate path: {path}")
        paths.add(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Changed audio: {path}")
        if digest in hashes and hashes[digest] != split:
            raise ValueError(f"Audio crosses splits: {path}")
        hashes[digest] = split
        engine = row.get("engine", "piper")
        if not row["speakers"]:
            raise ValueError(f"Missing voice identity: {path}")
        for voice in row["speakers"]:
            key = (engine, str(voice))
            if key in voices and voices[key] != split:
                raise ValueError(f"Voice crosses splits: {key}")
            voices[key] = split
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames != row["samples"]:
            raise ValueError(f"Unexpected audio format/length: {path}")
        if row["label"] and not 0 < row["keyword_end"] <= info.frames:
            raise ValueError(f"Invalid keyword boundary: {path}")
        counts[f"{split}/{row['label']}"] += 1
        seconds[split] += info.duration
    if any(not counts[f"{s}/{y}"] for s in ("training", "validation", "testing") for y in (0, 1)):
        raise ValueError("Each split must contain both labels")
    return {"clips": len(rows), "counts": dict(counts), "seconds": dict(seconds),
            "voice_ids": len(voices), "voice_and_audio_disjoint": True,
            "manifest_sha256": hashlib.sha256((root / "manifest.jsonl").read_bytes()).hexdigest(),
            "limits": "Voice IDs are engine presets; this does not establish independent human speakers or pronunciation quality."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
