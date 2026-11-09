"""Prepare auditable LibriSpeech continuous negatives for a frozen detector.

The corpus must already be extracted from the checksum-verified archive.
These are evaluation-only features, never automatically added to training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np
import soundfile as sf

from features import continuous_wav_features

ALIASES = re.compile(r"\b(?:nyx|nix|nicks|esp)\b|\bhi\s+e\s+s\s+p\b", re.I)


def worker(job):
    speaker, records, root = job
    root = Path(root)
    sources = []
    speech_samples = 0
    wav_path = root/"pcm"/f"{speaker}.wav"
    feature_path = root/"features"/f"{speaker}.npy"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        for record in records:
            path = Path(record["path"])
            pcm, rate = sf.read(path, dtype="int16")
            if rate != 16000 or pcm.ndim != 1:
                raise ValueError("Expected 16 kHz mono speech")
            speech_samples += len(pcm)
            # Count every separator in the actual WAV sent to both detectors.
            joined = np.concatenate((pcm, np.zeros(3200, dtype=np.int16)))
            wav.writeframes(joined.astype("<i2").tobytes())
            sources.append({"utterance": record["utterance"], "samples": len(pcm),
                "flac_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "transcript_sha256": record["transcript_sha256"]})
    raw, padding = continuous_wav_features(wav_path)
    np.save(feature_path, raw)
    return {"speaker": speaker, "split": "evaluation_only", "path": str(feature_path.relative_to(root)),
            "pcm_path": str(wav_path.relative_to(root)), "frames": len(raw),
            "seconds": (speech_samples+3200*len(records))/16000,
            "speech_seconds": speech_samples/16000, "separator_seconds": .2*len(records),
            "final_feature_padding_samples": padding,
            "feature_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "pcm_sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(), "sources": sources}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    with args.archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            md5.update(block)
            sha256.update(block)
    if md5.hexdigest() != args.expected_md5:
        raise ValueError("Archive checksum mismatch")
    groups, excluded = {}, []
    for path in sorted(args.corpus.rglob("*.trans.txt")):
        for line in path.read_text().splitlines():
            utterance, text = line.split(" ", 1)
            flac = path.parent/f"{utterance}.flac"
            if not flac.is_file():
                raise ValueError("Transcript has no corresponding FLAC")
            if ALIASES.search(text):
                excluded.append({"utterance": utterance, "reason": "wake_alias_in_transcript"})
                continue
            speaker = utterance.split("-")[0]
            groups.setdefault(speaker, []).append({"path": str(flac.resolve()), "utterance": utterance,
                "transcript_sha256": hashlib.sha256(text.encode()).hexdigest()})
    if not groups:
        raise ValueError("No eligible speech")
    args.output.mkdir(parents=True)
    (args.output/"pcm").mkdir()
    (args.output/"features").mkdir()
    records = []
    jobs = [(speaker, rows, str(args.output)) for speaker, rows in sorted(groups.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for record in pool.map(worker, jobs):
            records.append(record)
            print(json.dumps({"speakers": len(records), "total": len(groups),
                              "hours": sum(r["seconds"] for r in records)/3600}), flush=True)
    report = {"dataset": "LibriSpeech test-clean", "license": "CC BY 4.0",
              "source": "https://www.openslr.org/12", "archive_md5": md5.hexdigest(),
              "archive_sha256": sha256.hexdigest(), "alias_pattern": ALIASES.pattern,
              "excluded": excluded, "records": records,
              "hours": sum(r["seconds"] for r in records)/3600,
              "speech_hours": sum(r["speech_seconds"] for r in records)/3600,
              "feature_method": "One frontend over exact WAV, whole-hop blocks, padding only the final incomplete 10 ms hop",
              "feature_code_sha256": hashlib.sha256(Path(__file__).with_name("features.py").read_bytes()).hexdigest(),
              "limits": ["Public audiobook speech, not household/TV/music or real-user positive acceptance.",
                         "Transcript alias screening cannot guarantee acoustic absence of a wake phrase.",
                         "Added 200 ms zero separators are counted and reported separately.",
                         "Evaluation-only source. If results inform model changes, later measurements are regressions."]}
    (args.output/"manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("hours", "speech_hours", "archive_sha256")}), flush=True)


if __name__ == "__main__":
    main()
