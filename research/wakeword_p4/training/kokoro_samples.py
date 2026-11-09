"""Add a local, independent synthesis engine with disjoint voice identities."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from kokoro_onnx import Kokoro
import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly
import soundfile as sf

from phrases import HARD_NEGATIVES, PARTIAL_WAKE_NEGATIVES

MODEL_SHA = "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a"
VOICES_SHA = "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partials-only", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "manifest.jsonl").exists(), "Use a fresh output directory"
    model = args.assets / "kokoro-v1.0.onnx"
    voices_path = args.assets / "voices-v1.0.bin"
    for path, expected in ((model, MODEL_SHA), (voices_path, VOICES_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Hash mismatch: {path.name}")
    class TypedSpeedSession(ort.InferenceSession):
        # kokoro-onnx 0.4.9 chooses int32 speed by input name, but this pinned
        # model declares a float tensor. Respect the actual ONNX schema.
        def run(self, output_names, input_feed, run_options=None):
            speed = next(i for i in self.get_inputs() if i.name == "speed")
            dtype = np.float32 if speed.type == "tensor(float)" else np.int32
            inputs = {**input_feed, "speed": np.asarray(input_feed["speed"], dtype=dtype)}
            return super().run(output_names, inputs, run_options)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = TypedSpeedSession(str(model), options, providers=["CPUExecutionProvider"])
    kokoro = Kokoro.from_session(session, str(voices_path))
    voices = sorted(v for v in kokoro.get_voices() if v.startswith(("af_", "am_", "bf_", "bm_")))
    # Stratify gender/accent, but hold out complete voice IDs before generation.
    rng = np.random.default_rng(726)
    splits = {s: [] for s in ("training", "validation", "testing")}
    for prefix in ("af_", "am_", "bf_", "bm_"):
        group = [v for v in voices if v.startswith(prefix)]
        rng.shuffle(group)
        splits["validation"].extend(group[:1])
        splits["testing"].extend(group[1:2])
        splits["training"].extend(group[2:])
    started = time.monotonic()
    count = 0
    with (root / "manifest.jsonl").open("w") as log:
        for split, ids in splits.items():
            for voice in ids:
                # Both spellings are positive; identical spellings are not a
                # substitute for source-engine and held-out-speaker diversity.
                texts = [("Hey Nicks.", 1), ("Hey Nix!", 1), ("Hey Nyx.", 1)]
                texts += [(text, 0) for text in HARD_NEGATIVES]
                if args.partials_only:
                    texts = [(text, 0) for text in PARTIAL_WAKE_NEGATIVES]
                for index, (text, label) in enumerate(texts):
                    pcm, rate = kokoro.create(text, voice=voice, speed=1.0,
                                              lang="en-gb" if voice.startswith("b") else "en-us")
                    divisor = math.gcd(rate, 16000)
                    pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
                    relative = Path(split) / voice / f"{index:03d}.wav"
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, pcm, 16000, subtype="PCM_16")
                    row = {"path": str(relative), "split": split, "text": text,
                           "label": label, "speakers": [voice], "engine": "kokoro",
                           "samples": len(pcm), "keyword_end": len(pcm) if label else None,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    log.write(json.dumps(row)+"\n"); log.flush()
                    count += 1
                print(json.dumps({"voice": voice, "split": split, "clips": count,
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
    (root / "generation.json").write_text(json.dumps({"engine": "kokoro-onnx 0.4.9",
        "model_sha256": MODEL_SHA, "voices_sha256": VOICES_SHA,
        "voice_partitions": splits, "clips": count, "partials_only": args.partials_only,
        "keyword_end": "End of synthesized standalone wake clip; no phone timing available.",
        "seconds": round(time.monotonic()-started, 1)}, indent=2)+"\n")


if __name__ == "__main__":
    main()
