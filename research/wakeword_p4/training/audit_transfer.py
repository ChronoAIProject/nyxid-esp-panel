"""Audit a pretrained 30/60 encoder before resetting its classifier for Nyx.

Keeps the original Hey Jarvis head only for the numerical audit. This does not
evaluate Hey Nyx recognition and produces no production firmware.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from scipy.signal import resample_poly
import soundfile as sf

from features import features
from transfer import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mac-controls", action="store_true",
                        help="Also generate Hey Jarvis positives and negatives with Mac system TTS")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import utils
    from microwakeword.layers import modes
    model, metadata = build_model(149, args.source_model, reuse_head=True)
    args.output.mkdir(parents=True)
    model.save_weights(args.output/"best.weights.h5")
    streaming = utils.to_streaming_inference(model, {"stride": 3, "spectrogram_length": 149},
                                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), dtype=np.float32))
    streaming.export(str(args.output/"saved_model"))
    shutil.copyfile(args.source_model, args.output/"original_hey_jarvis.tflite")
    report = {"architecture": "transfer60", "architecture_args": ["--stride", "3",
        "--residual_connection", "0,0,0,0", "--pointwise_filters", "60,60,60,60",
        "--first_conv_filters", "30", "--first_conv_kernel_size", "5"],
        "input_frames": 149, "model_sha256": hashlib.sha256(args.source_model.read_bytes()).hexdigest(),
        "transfer": metadata, "scope": "Original Hey Jarvis head for import diagnostic only; not a Nyx candidate"}
    (args.output/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    subprocess.run([sys.executable, str(Path(__file__).with_name("audit_export.py")),
        "--upstream", str(args.upstream), "--features", str(args.features), "--model", str(args.output),
        "--tflite", str(args.output/"original_hey_jarvis.tflite"),
        "--output", str(args.output/"audit.json"), "--per-group", "4"], check=True)
    if args.mac_controls:
        controls = args.output/"controls"
        controls.mkdir()
        rows, offset = [], 0
        with (controls/"features.bin").open("wb") as handle:
            for voice in ("Daniel", "Samantha", "Moira"):
                for index, text in enumerate(("Hey Jarvis.", "Hey Nicks.", "Hey Mix.")):
                    path = controls/f"{voice}-{index}.aiff"
                    subprocess.run(["say", "-v", voice, "-o", str(path), text], check=True, timeout=30)
                    pcm, rate = sf.read(path, dtype="float32")
                    divisor = math.gcd(rate, 16000)
                    pcm = resample_poly(pcm, 16000//divisor, rate//divisor)
                    raw = features(np.concatenate((np.zeros(32000), pcm*32767, np.zeros(16000))))
                    handle.write(raw.astype("<u2").tobytes())
                    rows.append({"engine": "Mac_"+voice, "label": int(index == 0), "text": text,
                        "offset": offset, "frames": len(raw), "split": "validation", "variant": 0,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                    offset += len(raw)
        (controls/"features.json").write_text(json.dumps(rows)+"\n")
        subprocess.run([sys.executable, str(Path(__file__).with_name("audit_export.py")),
            "--upstream", str(args.upstream), "--features", str(controls), "--model", str(args.output),
            "--tflite", str(args.output/"original_hey_jarvis.tflite"),
            "--output", str(args.output/"audit-controls.json"), "--per-group", "4"], check=True)


if __name__ == "__main__":
    main()
