#!/usr/bin/env python3
"""Generate public test vectors; this does not train a usable wake-word model."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import wave

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from pymicro_features import MicroFrontend

MODEL_REVISION = "40ff33f57f8fc6ad71a75ef085abbf742495b225"
MODEL_SHA256 = "0689abe1912a95a3318a0d8cb2e67bad0cbcfe3e24dd6e050c75debddfb6f891"


def frontend_features(pcm):
    frontend = MicroFrontend()
    features = []
    for at in range(0, len(pcm), 160):
        result = frontend.process_samples(pcm[at:at + 160].tobytes())
        assert result.samples_read == 160
        if result.features:
            features.append(np.rint(np.array(result.features) * 25.6).astype(np.uint16))
    return np.array(features)


def quantize(raw):
    return np.clip((raw.astype(np.int32) * 256 + 333) // 666 - 128, -128, 127).astype(np.int8)


def declaration(name, dtype, values):
    flat = np.asarray(values).reshape(-1)
    rows = [", ".join(map(str, flat[i:i + 24].tolist())) for i in range(0, len(flat), 24)]
    return f"static const {dtype} {name}[] = {{\n" + ",\n".join(rows) + "\n};\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-model", type=Path, required=True,
                        help="Random-data compatibility model from the native Mac smoke check")
    parser.add_argument("--speech-dir", type=Path,
                        help="Optional synthetic Okay Nabu WAVs, 16 kHz mono PCM16, for high output scores")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "generated"
    root.mkdir(exist_ok=True)
    url = f"https://raw.githubusercontent.com/esphome/micro-wake-word-models/{MODEL_REVISION}/models/v2/okay_nabu.tflite"
    with urllib.request.urlopen(url, timeout=30) as response:
        (root / "okay_nabu.tflite").write_bytes(response.read())
    assert hashlib.sha256((root / "okay_nabu.tflite").read_bytes()).hexdigest() == MODEL_SHA256
    shutil.copyfile(args.smoke_model, root / "smoke.tflite")
    rng = np.random.default_rng(726)
    t = np.arange(32000) / 16000
    pcm = np.clip(4000 * np.sin(2 * np.pi * (180 * t + 1200 * t**2)) +
                  rng.normal(0, 300, len(t)), -32768, 32767).astype(np.int16)
    raw = frontend_features(pcm)
    assert raw.shape == (198, 40), raw.shape
    frames = rng.integers(-128, 128, size=(256, 1, 3, 40), dtype=np.int8)
    # Begin with real frontend values, then exercise the rest of the INT8 range.
    quantized = quantize(raw)
    for i in range(len(raw) // 3):
        frames[i, 0] = quantized[i * 3:i * 3 + 3]
    speech_manifest = []
    if args.speech_dir:
        chunks = []
        for path in sorted(args.speech_dir.glob("*.wav")):
            with wave.open(str(path)) as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, "NONE")
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            chunks.extend((np.zeros(8000, dtype=np.int16), samples, np.zeros(8000, dtype=np.int16)))
            speech_manifest.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "samples": len(samples)})
        assert chunks, "No speech WAVs found"
        speech = np.concatenate(chunks)
        speech = np.pad(speech, (0, (-len(speech)) % 160))
        speech_features = quantize(frontend_features(speech))
        usable = len(speech_features) // 3 * 3
        frames = np.concatenate((frames, speech_features[:usable].reshape(-1, 1, 3, 40)))
    header = "#pragma once\n#include <stdint.h>\n"
    header += f"static constexpr unsigned test_step_count = {len(frames)};\n"
    header += declaration("test_pcm", "int16_t", pcm)
    header += declaration("expected_features", "uint16_t", raw)
    header += declaration("test_frames", "int8_t", frames)
    manifest = {"model_revision": MODEL_REVISION, "frames": len(frames),
                "frontend_frames": len(raw), "models": {},
                "synthetic_speech_fixtures": speech_manifest,
                "host_kernel": "BUILTIN_REF (XNNPACK disabled)",
                "scope": "Numeric/streaming compatibility only; no wake-word accuracy evidence"}
    for name in ("okay_nabu", "smoke"):
        model = root / (name + ".tflite")
        # Use canonical reference kernels. XNNPACK's INT8 rounding differs from
        # TFLM by up to 10 output units on the high-score speech fixtures.
        interpreter = Interpreter(model_path=str(model), num_threads=1,
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        interpreter.allocate_tensors()
        inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        assert inp["shape"].tolist() == [1, 3, 40] and inp["dtype"] == np.int8
        assert output["dtype"] == np.uint8
        scores = []
        for frame in frames:
            interpreter.set_tensor(inp["index"], frame)
            interpreter.invoke()
            scores.append(int(interpreter.get_tensor(output["index"])[0, 0]))
        header += declaration("expected_" + name, "uint8_t", scores)
        manifest["models"][name] = {"sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                                    "bytes": model.stat().st_size,
                                    "score_min": min(scores), "score_max": max(scores)}
    (root / "vectors.h").write_text(header)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
