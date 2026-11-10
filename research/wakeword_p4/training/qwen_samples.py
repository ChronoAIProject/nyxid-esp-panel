"""Pinned CUDA Qwen VoiceDesign synthesis; description conditions are training-only."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from phrases import HARD_NEGATIVES


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profiles", type=int, default=24)
    p.add_argument("--clips-per-profile", type=int, default=15)
    a = p.parse_args()
    if not 1 <= a.profiles <= 96 or not 1 <= a.clips_per_profile <= 15:
        p.error("profiles must be 1..96; clips-per-profile must be 1..15")
    if a.output.exists():
        p.error("Use a fresh output directory; partial runs are preserved, never overwritten")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    source = json.loads(a.source.read_text())
    if source != {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                  "revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3"}:
        raise ValueError("Unexpected model source")
    conditions = list(itertools.product(
        ("young adult", "middle-aged", "older adult"), ("female", "male"),
        ("American", "British", "Australian", "Irish"),
        ("warm clear", "soft breathy", "low resonant", "bright slightly nasal")))
    np.random.default_rng(726).shuffle(conditions)
    texts = [("Hey Nicks.", 1), ("Hey Nix!", 1), ("Hey Nicks?", 1)]
    texts += [(text, 0) for text in HARD_NEGATIVES[:8] + HARD_NEGATIVES[-4:]]
    plan = []
    for profile, (age, gender, accent, texture) in enumerate(conditions[:a.profiles]):
        description = (f"A {age} {gender} voice, {texture}, speaking English with "
                       f"an {accent} accent. Natural conversational delivery.")
        for number, (text, label) in enumerate(texts[:a.clips_per_profile]):
            plan.append({"path": f"training/profile_{profile:03d}/{number:03d}.wav",
                         "split": "training", "speakers": [f"condition_{profile:03d}"],
                         "description": description, "text": text, "label": label,
                         "seed": 726 + profile * 100 + number, "engine": "qwen-cuda"})
    a.output.mkdir(parents=True)
    (a.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    generation = dict(max_new_tokens=96, do_sample=True, top_k=50, top_p=1.0,
                      temperature=0.8, repetition_penalty=1.05,
                      subtalker_dosample=True, subtalker_top_k=50,
                      subtalker_top_p=1.0, subtalker_temperature=0.8)
    report = {"source": source, "model_files": {
        str(path.relative_to(a.model)): sha(path) for path in sorted(a.model.rglob("*"))
        if path.is_file() and ".cache" not in path.parts},
        "source_sha256": sha(Path(__file__)), "plan_sha256": sha(a.output / "plan.json"),
        "runtime": {name: version(name) for name in
                    ("torch", "torchaudio", "qwen-tts", "transformers", "numpy", "scipy")},
        "gpu": torch.cuda.get_device_name(0), "dtype": "bfloat16", "attention": "sdpa",
        "generation": generation, "planned": len(plan),
        "limits": ["Descriptions and seeds are not independent human speaker identities; training-only.",
                   "Waveform QA does not establish word correctness; independent review is required.",
                   "Seeded GPU sampling is not claimed bit-exact across runtimes.",
                   "keyword_end is clip end, not forced-aligned phonetic timing."]}
    (a.output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.monotonic()
    model = Qwen3TTSModel.from_pretrained(str(a.model.resolve()), device_map="cuda:0",
                                         dtype=torch.bfloat16, attn_implementation="sdpa")
    accepted, rejected = 0, []
    with (a.output / "manifest.jsonl").open("w") as log:
        for row in plan:
            torch.manual_seed(row["seed"])
            torch.cuda.manual_seed_all(row["seed"])
            wavs, rate = model.generate_voice_design(text=row["text"], language="English",
                                                      instruct=row["description"], **generation)
            if len(wavs) != 1:
                raise ValueError("Unexpected batch size")
            pcm = np.asarray(wavs[0], dtype=np.float32)
            if pcm.ndim != 1 or not np.isfinite(pcm).all() or not len(pcm):
                raise ValueError("Invalid synthesized PCM")
            original_samples = len(pcm)
            divisor = math.gcd(rate, 16000)
            pcm = resample_poly(pcm, 16000 // divisor, rate // divisor)
            duration = len(pcm) / 16000
            rms = float(np.sqrt(np.mean(pcm ** 2)))
            clipping = float(np.mean(np.abs(pcm) >= .999))
            qa = {"duration": duration, "rms": rms, "clipping": clipping}
            # Token-cap-length output may be truncated; exclude rather than label it correct.
            if not (.25 <= duration <= 5 and rms >= .003 and clipping < .01):
                rejected.append({**row, **qa})
            else:
                path = a.output / row["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, pcm, 16000, subtype="PCM_16")
                log.write(json.dumps({**row, **qa, "samples": len(pcm),
                    "model_sample_rate": rate, "model_samples": original_samples,
                    "keyword_end": len(pcm) if row["label"] else None,
                    "sha256": sha(path)}) + "\n")
                log.flush()
                accepted += 1
            print(json.dumps({"accepted": accepted, "rejected": len(rejected),
                              "planned": len(plan), "seconds": round(time.monotonic()-started, 1)}),
                  flush=True)
    report.update(accepted=accepted, rejected=rejected,
                  manifest_sha256=sha(a.output / "manifest.jsonl"),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                  seconds=round(time.monotonic()-started, 1))
    (a.output / "generation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
