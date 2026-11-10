"""Independent ASR consistency screen; never filter or relabel holdout clips."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from faster_whisper import WhisperModel
import soundfile as sf


def normalized(text):
    return " ".join(re.findall(r"[a-z]+", text.lower()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-description", default="faster-whisper 1.2.1 tiny.en CPU INT8, beam=3, no prompt/VAD")
    p.add_argument("--model-source", type=Path, required=True)
    p.add_argument("--details", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--training-source", action="store_true",
                   help="Audit a training-only synthesis manifest instead of the paired challenge")
    a = p.parse_args()
    if a.details.exists() or a.output.exists():
        p.error("Use fresh audit outputs")
    model = WhisperModel(str(a.model), device="cpu", compute_type="int8", cpu_threads=4)
    counts, by_voice = Counter(), defaultdict(Counter)
    rows = list(map(json.loads, (a.dataset/"manifest.jsonl").read_text().splitlines()))
    if a.training_source and any(r["split"] != "training" for r in rows):
        raise ValueError("Training-source mode requires exclusively training clips")
    a.details.parent.mkdir(parents=True, exist_ok=True)
    with a.details.open("w") as log:
        for index, row in enumerate(rows, 1):
            target = ("hey_nyx" if row["label"] else "negative") if a.training_source else row["target"]
            voice = "/".join(map(str, row["speakers"])) if a.training_source else row["voice"]
            path = a.dataset/row["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Frozen audio changed")
            pcm, rate = sf.read(path, dtype="float32")
            if rate != 16000 or pcm.ndim != 1:
                raise ValueError("Expected mono 16 kHz PCM")
            segments, _ = model.transcribe(pcm, language="en", beam_size=3,
                                           condition_on_previous_text=False,
                                           temperature=0, vad_filter=False)
            text = " ".join(s.text for s in segments).strip()
            words = normalized(text)
            expected = normalized(row["text"])
            aliases = {expected}
            if target == "hey_nyx":
                aliases |= {"hey nicks", "hey nix", "hey nyx", "hey knicks", "hey nick s"}
            if target == "hi_esp":
                aliases |= {"hi esp", "hi e s p"}
            match = words in aliases
            target = str(target)
            counts[f"{target}/clips"] += 1
            counts[f"{target}/text_agreements"] += int(match)
            by_voice[voice]["clips"] += 1
            by_voice[voice]["text_agreements"] += int(match)
            log.write(json.dumps({"path": row["path"], "sha256": row["sha256"],
                                  "expected": row["text"], "transcript": text,
                                  "text_agreement": match, "target": target})+"\n")
            log.flush()
            if index % 50 == 0:
                print(json.dumps({"reviewed": index, "total": len(rows)}), flush=True)
    files = {}
    for path in sorted(a.model.iterdir()):
        if path.is_file():
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"asr": a.model_description,
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "asr_source": json.loads(a.model_source.read_text()), "asr_files_sha256": files,
              "manifest_sha256": hashlib.sha256((a.dataset/"manifest.jsonl").read_bytes()).hexdigest(),
              "clips": len(rows), "counts": dict(counts),
              "training_source": a.training_source,
              "by_voice": {v: dict(c) for v,c in by_voice.items()},
              "clips_removed_or_relabeled": 0, "wake_classifier_scores_computed": False,
              "limits": "Text agreement is not proof of phonetic accuracy; mismatches may be ASR errors. All frozen WAVs and labels are retained unchanged. This screen is not wake recognition, human listening review, or Hi ESP evaluation."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
