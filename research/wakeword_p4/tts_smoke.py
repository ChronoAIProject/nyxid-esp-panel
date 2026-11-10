#!/usr/bin/env python3
"""Check pinned native Piper sample generation; not a wake-word accuracy test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import wave

import numpy as np
import torch
from piper_phonemize import phonemize_espeak

REVISION = "b5b36e266576693ea47db129fbfe6bbc06164762"
CHECKPOINT_SHA256 = "e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phrase", default="Hey Nyx.")
    args = parser.parse_args()
    source = args.upstream.resolve()
    destination = args.output_dir.resolve()
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == REVISION
    checkpoint = source / "models/en_US-libritts_r-medium.pt"
    # The upstream checkpoint is a pickled full model, not just weights. Only
    # deserialize the exact release used here; the small torch 2.8 patch is explicit.
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    assert "map_location=\"cpu\", weights_only=False" in (source / "generate_samples.py").read_text()
    assert not destination.exists() or not any(destination.iterdir()), "Use an empty output directory"
    phonemes = {s: phonemize_espeak(s, "en-us") for s in ("Hey Nyx.", "Hey Nicks.", "Hey Nix.")}
    assert phonemes["Hey Nyx."] == phonemes["Hey Nicks."] == phonemes["Hey Nix."]
    sys.path.insert(0, str(source))
    from generate_samples import generate_samples

    torch.set_num_threads(4)
    torch.manual_seed(726)
    started = time.monotonic()
    generate_samples(args.phrase, destination, max_samples=12, batch_size=4,
                     max_speakers=904, length_scales=(1.0,), model=checkpoint)
    samples = []
    for path in sorted(destination.glob("*.wav"), key=lambda p: int(p.stem)):
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, "NONE")
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        peak = int(np.max(np.abs(pcm.astype(np.int32))))
        assert peak > 1000 and len(pcm) > 1600
        samples.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "seconds": round(len(pcm) / 16000, 3), "peak": peak})
    assert len(samples) == 12
    report = {"phrase": args.phrase, "pronunciation": "Hey Nicks", "generator_revision": REVISION,
              "checkpoint_sha256": CHECKPOINT_SHA256, "torch": torch.__version__,
              "mps_available": torch.backends.mps.is_available(), "elapsed_seconds": round(time.monotonic() - started, 2),
              "phonemes": phonemes, "samples": samples,
              "scope": "Twelve synthetic samples only; no wake accuracy, human pronunciation review, or speaker holdout evidence"}
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "phonemes")}, indent=2))


if __name__ == "__main__":
    main()
