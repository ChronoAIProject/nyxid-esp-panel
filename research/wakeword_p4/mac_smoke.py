"""Native macOS compatibility check; random data, NOT a usable wake-word model."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--upstream", type=Path, required=True, help="Pinned microWakeWord checkout")
cli.add_argument("--output-dir", type=Path, required=True)
args = cli.parse_args()
ROOT = args.output_dir.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.upstream.resolve()))
report = {"platform": platform.platform(), "python": sys.version, "steps": {}}
started = time.monotonic()


def save():
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (ROOT / "smoke-result.json").write_text(json.dumps(report, indent=2) + "\n")


def record(name, value):
    report["steps"][name] = value
    print(name, json.dumps(value), flush=True)
    save()


try:
    import numpy as np
    import tensorflow as tf
    from pymicro_features import MicroFrontend
    from ai_edge_litert.interpreter import Interpreter

    record("imports", {"tensorflow": tf.__version__, "numpy": np.__version__})
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(42)
    frontend = MicroFrontend()
    audio = (1000 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype(np.int16)
    frames = []
    for index in range(0, len(audio), 160):
        result = frontend.process_samples(audio[index:index + 160].tobytes())
        if result.features:
            frames.append(result.features)
    assert len(frames) > 0 and len(frames[0]) == 40
    record("native_frontend", {"feature_frames": len(frames), "features_per_frame": len(frames[0])})

    from microwakeword import mixednet, utils
    from microwakeword.layers import modes

    parser = argparse.ArgumentParser()
    mixednet.model_parameters(parser)
    flags = parser.parse_args(["--residual_connection", "0,0,0,0", "--stride", "3"])
    model = mixednet.model(flags, (180, 40), None)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    rng = np.random.default_rng(42)
    samples = rng.uniform(0, 26, size=(8, 180, 40)).astype(np.float32)
    labels = np.array([0, 1] * 4, dtype=np.float32).reshape(-1, 1)
    losses = [float(model.train_on_batch(samples, labels)) for _ in range(3)]
    record("upstream_model_training", {"parameters": model.count_params(), "batches": 3, "losses": losses, "synthetic_random_data_only": True})

    streaming = utils.to_streaming_inference(model, {"stride": 3, "spectrogram_length": 180}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), dtype=np.float32))
    record("upstream_streaming_conversion", {"input_shape": list(streaming.input_shape), "output_shape": list(streaming.output_shape)})
    streaming.export(str(ROOT / "smoke-saved-model"))

    def calibration():
        for _ in range(100):
            yield [rng.uniform(0, 26, size=(1, 3, 40)).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(ROOT / "smoke-saved-model"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = calibration
    flatbuffer = converter.convert()
    (ROOT / "smoke-streaming-int8.tflite").write_bytes(flatbuffer)
    record("int8_export", {"model_bytes": len(flatbuffer)})
    interpreter = Interpreter(model_content=flatbuffer, num_threads=2)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    for _ in range(10):
        interpreter.set_tensor(input_details["index"], np.zeros(input_details["shape"], dtype=input_details["dtype"]))
        interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])
    record("native_quantized_streaming_inference", {"calls": 10, "input_dtype": str(input_details["dtype"]), "output_dtype": str(output_details["dtype"]), "output": output.tolist()})
    report["result"] = "PASS: compatibility only; no wake-word accuracy or ESP32-P4 validation"
except Exception as exc:
    import traceback
    report["result"] = "FAIL"
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
    save()
    raise
save()
