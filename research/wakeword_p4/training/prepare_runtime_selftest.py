"""Make a private exact-PCM/INT8-output fixture for a full firmware diagnostic.

This is a numeric regression, not microphone injection or recognition acceptance.
The detector is destroyed after self-test, before normal microphone listening.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import soundfile as sf

from evaluate import Detector
from features import features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Use fresh private output paths")
    pcm, rate = sf.read(args.audio, dtype="int16")
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError("Expected original 16 kHz mono PCM; do not change speech speed")
    # Bound firmware flash use and retain complete detector input chunks.
    samples = min(len(pcm), 4 * 16000) // 480 * 480
    if samples < 960:
        raise ValueError("Audio is too short")
    pcm = np.asarray(pcm[:samples], dtype="<i2")
    detector = Detector(args.model)
    scores = detector.predict(features(pcm))
    expected = np.rint(scores / detector.output_scale + detector.output_zero)
    if (len(expected) != samples // 480 - 1 or not np.isfinite(expected).all()
            or expected.min() < 0 or expected.max() > 255 or expected.min() == expected.max()):
        raise ValueError("Uninformative or invalid diagnostic outputs")
    digest = hashlib.sha256(args.model.read_bytes()).digest()
    data = (b"NYXKWS01" + digest + struct.pack("<II", samples, len(expected))
            + pcm.tobytes() + expected.astype(np.uint8).tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    report = {"scope": "Known-PCM numeric regression inside the complete firmware; not acoustic acceptance",
              "model_sha256": digest.hex(), "fixture_sha256": hashlib.sha256(data).hexdigest(),
              "source_audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              "sample_rate": rate, "samples": samples, "chunk_samples": 480,
              "expected_outputs": len(expected), "minimum_output": int(expected.min()),
              "maximum_output": int(expected.max()),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
