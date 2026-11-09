"""Prepare new, training-only human negatives from LibriSpeech train-clean-100.

Verify the archive before extraction, keep speaker IDs disjoint from all
existing evaluation speakers, screen wake aliases, and preserve provenance.
This command does not read evaluation scores or change existing artifacts.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from continuous_negatives import worker

EXPECTED_MD5 = "2a93770f6d5c6c964bc36631d331a522"
ALIASES = re.compile(r"\b(?:nyx|nix|nicks|nick['’]s|esp)\b|\bhi\s+e\s+s\s+p\b", re.I)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--existing-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-speakers", type=int, default=64,
                        help="Select this many speaker IDs before feature extraction; zero selects all")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_speakers < 0:
        parser.error("Speaker limit cannot be negative")
    if args.output.exists():
        parser.error("Use a fresh output directory")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with args.archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            md5.update(block); sha256.update(block)
    if md5.hexdigest() != EXPECTED_MD5:
        raise ValueError("Official train-clean-100 archive checksum mismatch")
    args.output.mkdir(parents=True)
    extracted = args.output/"source"
    extracted.mkdir()
    with tarfile.open(args.archive, "r:gz") as archive:
        archive.extractall(extracted, filter="data")
    existing = json.loads((args.existing_features/"human.json").read_text())
    forbidden = {str(r["speaker"]) for r in existing}
    groups, excluded = {}, []
    for path in sorted((extracted/"LibriSpeech/train-clean-100").rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            utterance, text = line.split(" ", 1)
            speaker = utterance.split("-")[0]
            if speaker in forbidden:
                raise ValueError("New training source overlaps an existing speaker partition")
            if ALIASES.search(text):
                excluded.append({"utterance": utterance, "reason": "wake_alias_in_transcript"})
                continue
            flac = path.parent/f"{utterance}.flac"
            if not flac.is_file():
                raise ValueError("Transcript has no corresponding audio")
            groups.setdefault(speaker, []).append({"path": str(flac.resolve()), "utterance": utterance,
                "transcript_sha256": hashlib.sha256(text.encode()).hexdigest()})
    speakers = sorted(groups)
    np.random.default_rng(726).shuffle(speakers)
    if args.max_speakers:
        speakers = speakers[:args.max_speakers]
    if not speakers:
        raise ValueError("No eligible training speaker")
    (args.output/"pcm").mkdir()
    (args.output/"features").mkdir()
    rows = []
    jobs = [(speaker, groups[speaker], str(args.output)) for speaker in sorted(speakers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(worker, jobs):
            rows.append({**record, "split": "training"})
            print(json.dumps({"speakers": len(rows), "total": len(jobs),
                              "hours": sum(r["seconds"] for r in rows)/3600}), flush=True)
    report = {"dataset": "LibriSpeech train-clean-100", "license": "CC BY 4.0",
              "source": "https://www.openslr.org/12", "archive_md5": md5.hexdigest(),
              "archive_sha256": sha256.hexdigest(), "records": rows, "excluded": excluded,
              "selection": "Deterministic speaker permutation (seed 726), before feature extraction; training only",
              "alias_pattern": ALIASES.pattern,
              "existing_speaker_index_sha256": hashlib.sha256((args.existing_features/"human.json").read_bytes()).hexdigest(),
              "hours": sum(r["seconds"] for r in rows)/3600,
              "speech_hours": sum(r["speech_seconds"] for r in rows)/3600,
              "feature_method": "One frontend over exact WAV; padding only the final partial hop",
              "limits": ["Transcript screening does not certify acoustic absence of every wake alias.",
                         "Audiobook speech does not replace household, television, Chinese or music recordings."]}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
