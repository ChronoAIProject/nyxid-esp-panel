"""Reconstruct pinned Hey Jarvis weights; audit on training features only.

This is a numerical reconstruction diagnostic, never Nyx recognition evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from evaluate import Detector

ARCH = ["--residual_connection", "0,0,0,0", "--stride", "3",
        "--pointwise_filters", "60,60,60,60", "--repeat_in_block", "1,1,1,1",
        "--mixconv_kernel_sizes", "[5], [9], [13], [21]",
        "--first_conv_filters", "30", "--first_conv_kernel_size", "5"]
SOURCE_SHA = "21a7976add39ee24ec96c63d96b7aaa18e24d1d9824b963e451da8feb4b78b77"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "model", "features", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    if hashlib.sha256(a.model.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected source model")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    sys.path.insert(0, str(a.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    parser = argparse.ArgumentParser(); mixednet.model_parameters(parser)
    model = mixednet.model(parser.parse_args(ARCH), (150, 40), None)
    interpreter = Interpreter(model_path=str(a.model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    interpreter.allocate_tensors()
    details = {d["index"]: d for d in interpreter.get_tensor_details()}
    ops = [op for op in interpreter._get_ops_details()
           if op["op_name"] in ("CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED")]
    def dequant(index):
        q = details[int(index)]["quantization_parameters"]
        value = interpreter.get_tensor(int(index)).astype(np.float32)
        shape = [1] * value.ndim
        shape[q["quantized_dimension"]] = len(q["scales"])
        return (value - q["zero_points"].reshape(shape)) * q["scales"].reshape(shape)
    cursor, pending_bias = 0, None
    for layer in model.layers:
        cell = getattr(layer, "cell", layer)
        if isinstance(cell, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D, tf.keras.layers.Dense)):
            op = ops[cursor]; cursor += 1
            expected = ("DEPTHWISE_CONV_2D" if isinstance(cell, tf.keras.layers.DepthwiseConv2D)
                        else "FULLY_CONNECTED" if isinstance(cell, tf.keras.layers.Dense) else "CONV_2D")
            if op["op_name"] != expected:
                raise ValueError("Unexpected operator order")
            kernel, bias = dequant(op["inputs"][1]), dequant(op["inputs"][2])
            kernel = kernel.T if expected == "FULLY_CONNECTED" else kernel.transpose(1, 2, 3, 0)
            if cell.use_bias:
                cell.set_weights([kernel, bias])
            else:
                cell.set_weights([kernel])
                if cursor == 1:
                    np.testing.assert_array_equal(bias, 0)
                else:
                    pending_bias = bias
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            if pending_bias is None:
                raise ValueError("No fused bias for batch normalization")
            # Reproduce fused affine weights exactly in inference mode. These
            # identity moving statistics are reconstructed, not original stats.
            ones = np.ones_like(pending_bias)
            layer.set_weights([ones * np.sqrt(1 + layer.epsilon), pending_bias,
                               np.zeros_like(ones), ones])
            layer.trainable = False
            pending_bias = None
    if cursor != len(ops) or pending_bias is not None:
        raise ValueError("Unmapped source operators")
    a.output.mkdir(parents=True)
    model.save_weights(a.output / "reconstructed.weights.h5")
    streaming = utils.to_streaming_inference(model, {"stride": 3, "spectrogram_length": 150},
                                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), np.float32))
    states = [v for v in streaming.variables if "states" in v.name]
    if len(states) != 6:
        raise ValueError("Unexpected streaming state buffers")
    @tf.function(input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
    def infer(x):
        return streaming(x, training=False)
    raw = np.memmap(a.features / "features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    rows = json.loads((a.features / "features.json").read_text())
    seen, results, deltas = {}, [], []
    detector = Detector(a.model)
    for row in rows:
        key = row["engine"], row["label"]
        if row["split"] != "training" or row["variant"] != 0 or seen.get(key, 0) >= 2:
            continue
        seen[key] = seen.get(key, 0) + 1
        data = raw[row["offset"]:row["offset"] + row["frames"]]
        values = np.clip(data.astype(np.float32) / 25.6, 0, 26)
        for state in states:
            state.assign(tf.zeros_like(state))
        actual = np.array([float(infer(values[None, at:at+3]).numpy()[0, 0])
                           for at in range(0, len(values)-2, 3)])
        detector.reset(); expected = detector.predict(data)
        difference = np.abs(actual - expected)
        if not np.isfinite(actual).all() or len(actual) != len(expected):
            raise ValueError("Invalid reconstructed predictions")
        deltas.extend(difference.tolist())
        results.append(dict(path=row["path"], engine=row["engine"], label=row["label"],
                            steps=len(actual), mean_absolute_error=float(difference.mean()),
                            max_absolute_error=float(difference.max()),
                            original_max=float(expected.max()), reconstructed_max=float(actual.max())))
    report = dict(role="Pinned Hey Jarvis numerical reconstruction on Nyx training features, not recognition evidence",
                  source_model_sha256=SOURCE_SHA, context_frames=150, architecture_args=ARCH,
                  parameters=model.count_params(), mapped_operators=cursor,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  feature_index_sha256=hashlib.sha256((a.features / "features.json").read_bytes()).hexdigest(),
                  mean_absolute_error=float(np.mean(deltas)), max_absolute_error=float(np.max(deltas)),
                  clips=results, validation_or_final_accessed=False,
                  limits="Dequantized folded weights; original batch-normalization statistics are unavailable. A small non-Jarvis subset cannot certify full parity or transfer quality. Requires target-word input probes and intermediate-activation checks before transfer training.")
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}), flush=True)


if __name__ == "__main__":
    main()
