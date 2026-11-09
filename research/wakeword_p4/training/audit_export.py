"""Compare aligned float windows, float streaming and INT8 on validation only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from evaluate import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tflite", type=Path, help="Explicit INT8 artifact; defaults to model/hey_nyx.tflite")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=6)
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    report = json.loads((args.model/"training.json").read_text())
    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    flags = architecture.parse_args(report.get("architecture_args", ["--residual_connection", "0,0,0,0", "--stride", "3"]))
    frames = report.get("input_frames", 180)
    stride = flags.stride
    effective_frames = (frames-flags.first_conv_kernel_size)//stride*stride+flags.first_conv_kernel_size
    if report.get("architecture") == "transfer60":
        from transfer import build_model
        model, _ = build_model(frames)
    else:
        model = mixednet.model(flags, (frames, 40), None)
    model.load_weights(args.model/"best.weights.h5")
    saved = tf.saved_model.load(str(args.model/"saved_model"))
    function = saved.signatures["serving_default"]
    initial = [v.numpy() for v in saved.variables]

    @tf.function(input_signature=[tf.TensorSpec([None, 40], tf.float32)])
    def scan(data):
        count = tf.shape(data)[0]//stride
        values = tf.TensorArray(tf.float32, size=count)
        def body(i, array):
            output = function(data[i*stride:(i+1)*stride][None])
            value = next(iter(output.values()))[0, 0]
            return i+1, array.write(i, value)
        _, values = tf.while_loop(lambda i, _: i<count, body, (0, values), parallel_iterations=1)
        return values.stack()

    @tf.function(input_signature=[tf.TensorSpec([None, frames, 40], tf.float32)])
    def windows(data):
        return model(data, training=False)

    records = json.loads((args.features/"features.json").read_text())
    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    selected = []
    for engine in sorted({r["engine"] for r in records}):
        for label in (0, 1):
            # Select distinct original clips, not just four correlated variants.
            group = [r for r in records if r["split"] == "validation" and r["engine"] == engine and r["label"] == label and r["variant"] == 0]
            selected.extend(group[:args.per_group])
    detector = Detector(args.tflite or args.model/"hey_nyx.tflite")
    details = []
    for row in selected:
        source = raw[row["offset"]:row["offset"]+row["frames"]]
        scaled = np.clip(source.astype(np.float32)/25.6, 0, 26)
        for variable, value in zip(saved.variables, initial):
            variable.assign(value)
        floating = scan(scaled).numpy()
        detector.reset()
        quantized = detector.predict(source)
        starts = stride*(np.arange(len(floating))+1)-effective_frames
        valid = (starts >= 0) & (starts+frames <= len(source))
        clips = np.stack([scaled[at:at+frames] for at in starts[valid]])
        windowed = windows(clips).numpy().reshape(-1)
        fs = floating[valid]
        qs = quantized[valid]
        details.append({"engine": row["engine"], "label": row["label"],
                        "source_sha256": row["sha256"], "compared_steps": len(fs),
                        "float_window_stream_max_delta": float(np.max(np.abs(windowed-fs))),
                        "float_stream_int8_max_delta": float(np.max(np.abs(fs-qs))),
                        "float_stream_int8_mean_delta": float(np.mean(np.abs(fs-qs))),
                        "float_max": float(fs.max()), "int8_max": float(qs.max())})
        print(json.dumps(details[-1]), flush=True)
    result = {"model_sha256": report["model_sha256"], "input_frames": frames,
              "effective_frames": effective_frames, "stride": stride, "details": details,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Selected validation clips only; aligned numerical diagnostic, not a recognition or real-user acceptance result."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
