"""Audit a frozen paired-phrase corpus against development voice/audio IDs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--development", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    dev_voices, dev_audio = set(), set()
    for root in a.development:
        for row in map(json.loads, (root/"manifest.jsonl").read_text().splitlines()):
            dev_voices.update(str(v) for v in row["speakers"])
            dev_audio.add(row["sha256"])
    report = json.loads((a.dataset/"generation.json").read_text())
    rows = list(map(json.loads, (a.dataset/"manifest.jsonl").read_text().splitlines()))
    digest = hashlib.sha256((a.dataset/"manifest.jsonl").read_bytes()).hexdigest()
    if digest != report["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    counts, audio, pairs, paths = Counter(), set(), Counter(), set()
    for row in rows:
        path = (a.dataset/row["path"]).resolve()
        if not path.is_relative_to(a.dataset.resolve()) or path in paths:
            raise ValueError("Duplicate or escaping path")
        paths.add(path)
        if row["voice"] in dev_voices:
            raise ValueError("Voice preset was used in development")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != row["sha256"] or sha in dev_audio:
            raise ValueError("Audio changed or duplicates development audio")
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16" or info.frames != row["samples"]:
            raise ValueError("Audio format/length mismatch")
        audio.add(sha)
        counts[str(row["target"])] += 1
        if row["target"]:
            pairs[(row["voice"], row["speed"], row["target"])] += 1
    for voice in report["voice_ids"]:
        for speed in (.9, 1., 1.1):
            if any(pairs[(voice,speed,target)] != 1 for target in ("hey_nyx", "hi_esp")):
                raise ValueError("Unpaired target phrase")
    result = {"manifest_sha256": digest, "clips": len(rows), "unique_audio_hashes": len(audio),
              "voice_presets": len(report["voice_ids"]), "target_counts": dict(counts),
              "development_voice_and_audio_disjoint": True, "paired_voice_speed_checks_passed": True,
              "scores_computed": False,
              "limits": report["limits"]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
