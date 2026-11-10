"""Make a private exact-PCM/INT8-output fixture for a full firmware diagnostic.

This is a numeric regression, not microphone injection or recognition acceptance.
The detector is destroyed after self-test, before normal microphone listening.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import struct

import numpy as np
import soundfile as sf
from ai_edge_litert import schema_py_generated as schema

from evaluate import Detector
from features import features


RUNTIME_OPS = frozenset({"CALL_ONCE", "VAR_HANDLE", "READ_VARIABLE", "RESHAPE", "CONCATENATION",
    "STRIDED_SLICE", "ASSIGN_VARIABLE", "CONV_2D", "DEPTHWISE_CONV_2D", "SPLIT_V",
    "FULLY_CONNECTED", "LOGISTIC", "QUANTIZE", "EXPAND_DIMS"})


def inspect_model(data, detector):
    model = schema.Model.GetRootAsModel(data, 0)
    names = {number: name for name, number in vars(schema.BuiltinOperator).items() if type(number) is int}
    ops = set()
    for i in range(model.SubgraphsLength()):
        graph = model.Subgraphs(i)
        for j in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(j).OpcodeIndex())
            ops.add(names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())])
    if not ops or ops - RUNTIME_OPS:
        raise ValueError(f"Unsupported operators in P4 runtime: {sorted(ops - RUNTIME_OPS)}")
    if (len(detector.interpreter.get_input_details()) != 1 or len(detector.interpreter.get_output_details()) != 1
            or tuple(detector.input["shape"]) != (1, 3, 40) or detector.input["dtype"] != np.int8
            or tuple(detector.output["shape"]) != (1, 1) or detector.output["dtype"] != np.uint8
            or not np.isfinite(detector.scale) or detector.scale <= 0
            or (detector.output_scale, detector.output_zero) != (1 / 256, 0)):
        raise ValueError("Model does not expose the reviewed 3x40 INT8 / UINT8 streaming interface")
    return {"operators": sorted(ops), "subgraphs": model.SubgraphsLength(),
            "input_shape": [1, 3, 40], "input_quantization": [detector.scale, detector.zero],
            "output_shape": [1, 1], "output_quantization": [detector.output_scale, detector.output_zero]}


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
    model = args.model.read_bytes()
    detector = Detector(args.model)
    interface = inspect_model(model, detector)
    raw = features(pcm)
    scores = detector.predict(raw)
    detector.reset()
    replay = features(pcm)
    np.testing.assert_array_equal(raw, replay)
    np.testing.assert_array_equal(scores, detector.predict(replay))
    expected = np.rint(scores / detector.output_scale + detector.output_zero)
    if (len(expected) != samples // 480 - 1 or not np.isfinite(expected).all()
            or expected.min() < 0 or expected.max() > 255 or expected.min() == expected.max()):
        raise ValueError("Uninformative or invalid diagnostic outputs")
    digest = hashlib.sha256(model).digest()
    data = (b"NYXKWS01" + digest + struct.pack("<II", samples, len(expected))
            + pcm.tobytes() + expected.astype(np.uint8).tobytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    report = {"scope": "Known-PCM numeric regression inside the complete firmware; not acoustic acceptance",
              "model_sha256": digest.hex(), "fixture_sha256": hashlib.sha256(data).hexdigest(),
              "source_audio_sha256": hashlib.sha256(args.audio.read_bytes()).hexdigest(),
              "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
              "expected_outputs_sha256": hashlib.sha256(expected.astype(np.uint8).tobytes()).hexdigest(),
              "sample_rate": rate, "samples": samples, "chunk_samples": 480,
              "expected_outputs": len(expected), "minimum_output": int(expected.min()),
              "maximum_output": int(expected.max()),
              "host_interpreter": "BUILTIN_REF", "ai_edge_litert": version("ai-edge-litert"),
              "pymicro_features": version("pymicro-features"), "reset_reproducible": True,
              "interface": interface,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
