"""Quarantine suspicious generated speech with a separate, offline ASR engine."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

from faster_whisper import WhisperModel
import numpy as np


def normalize(text):
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation", "testing"), action="append",
                        help="Only inspect these splits; omit for legacy all-split behavior")
    parser.add_argument("--limit", type=int, default=0,
                        help="Deterministic balanced pilot; zero reviews the complete corpus")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(str(args.model), device="cpu", compute_type="int8", cpu_threads=3, num_workers=1)
    for source in args.source:
        assert (source/"generation.json").exists(), f"Generation incomplete: {source.name}"
        root = args.output/source.name
        root.mkdir(exist_ok=True)
        previous = {}
        if (root/"audit.jsonl").exists():
            previous = {r["path"]: r for r in map(json.loads, (root/"audit.jsonl").read_text().splitlines())}
        rows = list(map(json.loads, (source/"manifest.jsonl").read_text().splitlines()))
        total = len(rows)
        if args.split:
            rows = [r for r in rows if r["split"] in args.split]
        if args.limit:
            rng = np.random.default_rng(726)
            selected = []
            for label in (1, 0):
                group = [r for r in rows if r["label"] == label]
                rng.shuffle(group)
                selected.extend(group[:max(1, args.limit//2)])
            rows = selected
        accepted = []
        with (root/"audit.jsonl").open("a") as log:
            for number, row in enumerate(rows, 1):
                path = source/row["path"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row["sha256"]:
                    raise ValueError(f"Audio hash mismatch: {row['path']}")
                audit = previous.get(row["path"])
                if audit is None:
                    # All fixtures are already 16 kHz mono PCM16; avoid a
                    # redundant container decoder/resampler and its API drift.
                    with wave.open(str(path), "rb") as wav:
                        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
                        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)/32768
                    segments, _ = model.transcribe(pcm, language="en", beam_size=3,
                                                   vad_filter=False, condition_on_previous_text=False,
                                                   temperature=0, word_timestamps=False)
                    transcript = " ".join(s.text for s in segments).strip()
                    words = normalize(transcript)
                    # Do not prompt/bias ASR with the desired wake phrase.
                    positive = words in ("hey nicks", "hey nix", "hey nyx", "hey knicks")
                    expected = normalize(row["text"])
                    keep = positive if row["label"] else (not positive and words == expected)
                    audit = {"path": row["path"], "sha256": digest, "transcript": transcript,
                             "expected": row["text"], "label": row["label"], "accepted": keep}
                    log.write(json.dumps(audit)+"\n"); log.flush()
                if audit["accepted"]:
                    target = root/row["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.symlink_to(path.resolve())
                    accepted.append(row)
                if number % 50 == 0:
                    print(json.dumps({"source": source.name, "reviewed": number, "total": len(rows),
                                      "retained": len(accepted)}), flush=True)
        (root/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in accepted))
        report = {"engine": "faster-whisper tiny.en (CPU INT8)", "source": source.name,
                  "reviewed": len(rows), "retained": len(accepted),
                  "source_clips": total, "pilot_only": bool(args.limit),
                  "inspected_splits": args.split or ["training", "validation", "testing"],
                  "retained_positive": sum(r["label"] for r in accepted),
                  "retained_negative": sum(not r["label"] for r in accepted),
                  "limits": "ASR is an imperfect consistency screen, not proof of correct phonetics or a human listening evaluation. Rejected clips are quarantined, never relabeled."}
        (root/"generation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
