"""Generate local Qwen/OmniVoice/MOSS samples; keep model/source provenance."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import mlx.core as mx
from mlx_audio.tts.utils import load_model
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

from phrases import HARD_NEGATIVES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("qwen", "omni", "moss"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--profiles", type=int, default=24)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"generation.json").exists(), "Use a fresh completed-run directory"
    completed = {}
    if (root/"manifest.jsonl").exists():
        completed = {r["path"]: r for r in map(json.loads, (root/"manifest.jsonl").read_text().splitlines())}
    model = load_model(str(args.model.resolve()))
    conditions = list(itertools.product(
        ("young adult", "middle-aged", "older adult"),
        ("female", "male"),
        ("American", "British", "Australian", "Irish"),
        ("warm clear", "soft breathy", "low resonant", "bright slightly nasal")))
    rng = np.random.default_rng(726)
    rng.shuffle(conditions)
    plan = []
    if args.engine == "moss":
        assert args.references, "MOSS requires generated reference audio"
        rows = list(map(json.loads, (args.references/"manifest.jsonl").read_text().splitlines()))
        references = {}
        for row in rows:
            # Use longer negative sentences as carriers, not two-word wake clips.
            if row["samples"] > 24000:
                references.setdefault((row["split"], tuple(row["speakers"])), row)
        for (split, speakers), ref in list(references.items())[:args.profiles]:
            for number, (text, label) in enumerate([("Hey Nicks.", 1), ("Hey Nix!", 1)] + [(s, 0) for s in HARD_NEGATIVES[:8]]):
                voice = "_".join(map(str, speakers))
                plan.append({"path": f"{split}/{voice}/{number:03d}.wav", "split": split,
                             "speakers": list(speakers), "text": text, "label": label,
                             "reference": str((args.references/ref["path"]).resolve()),
                             "reference_text": ref["text"]})
    else:
        # No provable fixed speaker identity for text-designed/random voices.
        # Use these sources only for training; do not claim speaker-disjoint
        # validation merely because instruction strings or seeds differ.
        for profile, (age, gender, accent, texture) in enumerate(conditions[:args.profiles]):
            description = f"A {age} {gender} voice, {texture}, speaking English with an {accent} accent. Natural conversational delivery."
            texts = [("Hey Nicks.", 1), ("Hey Nix!", 1), ("Hey Nicks?", 1)]
            texts += [(text, 0) for text in HARD_NEGATIVES[:8]+HARD_NEGATIVES[-4:]]
            for number, (text, label) in enumerate(texts):
                plan.append({"path": f"training/profile_{profile:03d}/{number:03d}.wav",
                             "split": "training", "speakers": [f"condition_{profile:03d}"],
                             "text": text, "label": label, "description": description})
    pending = [r for r in plan if r["path"] not in completed]
    started = time.monotonic()
    rejected = []
    with (root/"manifest.jsonl").open("a") as log:
        for at in range(0, len(pending), args.batch if args.engine != "moss" else 1):
            batch = pending[at:at+(args.batch if args.engine != "moss" else 1)]
            seed = 726+at
            mx.random.seed(seed)
            if args.engine == "qwen":
                results = list(model.batch_generate(texts=[r["text"] for r in batch],
                    instructs=[r["description"] for r in batch], lang_code="English",
                    max_tokens=96, temperature=.8, stream=False, verbose=False))
            elif args.engine == "omni":
                results = model.generate_batch(text=[r["text"] for r in batch],
                    language="english", instruct=[r["description"] for r in batch],
                    duration_s=[1.0 if len(r["text"]) < 18 else 2.3 for r in batch],
                    num_steps=32, max_batch_size=args.batch)
            else:
                row = batch[0]
                results = list(model.generate(text=row["text"], ref_audio=row["reference"],
                    ref_text=row["reference_text"], max_tokens=96,
                    audio_tokenizer_source=str(args.codec.resolve()) if args.codec else None))
            for number, result in enumerate(results):
                index = getattr(result, "sequence_idx", number)
                item = batch[index]
                array = np.asarray(result.audio).astype(np.float32)
                original_shape = list(array.shape)
                array = np.squeeze(array)
                expected_samples = int(result.samples)
                if array.ndim == 1:
                    pcm = array
                elif array.ndim == 2 and array.shape[0] == expected_samples and array.shape[1] in (1, 2):
                    pcm = array.mean(axis=1)
                elif array.ndim == 2 and array.shape[1] == expected_samples and array.shape[0] in (1, 2):
                    pcm = array.mean(axis=0)
                else:
                    raise ValueError(f"Unsupported channel/sample shape: {original_shape}")
                if len(pcm) != expected_samples or not np.isfinite(pcm).all():
                    raise ValueError(f"Model sample count/content mismatch: {original_shape}, {expected_samples}")
                rate = result.sample_rate
                divisor = math.gcd(rate, 16000)
                pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
                duration = len(pcm)/16000
                rms = float(np.sqrt(np.mean(pcm**2))) if len(pcm) else 0
                clipping = float(np.mean(np.abs(pcm) >= .999)) if len(pcm) else 1
                if not (.25 <= duration <= 5 and rms >= .003 and clipping < .01):
                    rejected.append({"path": item["path"], "duration": duration,
                                     "rms": rms, "clipping": clipping})
                    continue
                path = root/item["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, pcm, 16000, subtype="PCM_16")
                # Private absolute carrier path stays out of generated metadata.
                reference = item.pop("reference", None)
                row = {**item, "engine": args.engine, "samples": len(pcm), "seed": seed,
                       "model_audio_shape": original_shape, "model_samples": expected_samples,
                       "keyword_end": len(pcm) if item["label"] else None,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                if reference:
                    row["reference_sha256"] = hashlib.sha256(Path(reference).read_bytes()).hexdigest()
                log.write(json.dumps(row)+"\n"); log.flush()
                completed[row["path"]] = row
            mx.clear_cache()
            print(json.dumps({"engine": args.engine, "accepted": len(completed),
                              "planned": len(plan), "rejected": len(rejected),
                              "seconds": round(time.monotonic()-started, 1)}), flush=True)
    model_files = {str(p.relative_to(args.model)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in args.model.rglob("*") if p.is_file() and ".cache" not in p.parts}
    report = {"engine": args.engine, "accepted": len(completed), "planned": len(plan),
              "rejected": rejected, "model_files": model_files,
              "limits": ["Waveform QA only; generated-word correctness still needs phonetic/ASR/listening review.",
                         "Qwen/Omni description conditions do not prove independent speaker IDs; these sources are training-only.",
                         "MOSS is a derivative of reference audio and must retain its source speaker split."]}
    (root/"generation.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
