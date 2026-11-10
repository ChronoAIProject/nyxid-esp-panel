"""Compare float SavedModel and INT8 streaming export on validation speech.

Quantization is expected to change scores. This reports drift and decision
changes, while requiring both paths to reproduce after explicit state reset.
It does not certify MCU compatibility or compare against WakeNet.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from evaluate import Detector, event_indices, smooth


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--threshold", type=float, required=True,
                   help="Previously frozen quantized validation threshold")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--per-group", type=int, default=2)
    a = p.parse_args()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    loaded = tf.saved_model.load(str(a.model_dir/"saved_model"))
    signature = loaded.signatures["serving_default"]
    keys = list(signature.structured_input_signature[1])
    if len(keys) != 1:
        raise ValueError("Expected one streaming feature input")
    states = [v for v in loaded.variables if "states" in v.name]
    if not states:
        raise ValueError("No float stream buffers found; cannot verify reset")
    def predict_float(raw):
        for state in states:
            state.assign(tf.zeros_like(state))
        x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        out = []
        for at in range(0, len(x)-2, 3):
            result = signature(**{keys[0]: tf.convert_to_tensor(x[None,at:at+3])})
            if len(result) != 1:
                raise ValueError("Expected one probability output")
            out.append(float(next(iter(result.values())).numpy().reshape(-1)[0]))
        return np.asarray(out)
    model = a.model_dir/"hey_nyx.tflite"
    quantized = Detector(model)
    data = np.memmap(a.features/"features.bin", mode="r", dtype="<u2").reshape(-1,40)
    rows = json.loads((a.features/"features.json").read_text())
    groups = Counter()
    results = []
    deltas = []
    for row in rows:
        if row["split"] != "validation" or row["variant"] != 0:
            continue
        key = (row["engine"], row["label"])
        if groups[key] >= a.per_group:
            continue
        groups[key] += 1
        raw = data[row["offset"]:row["offset"]+row["frames"]]
        f = predict_float(raw)
        f_reset = predict_float(raw)
        np.testing.assert_allclose(f_reset, f, rtol=0, atol=1e-7)
        quantized.reset(); q = quantized.predict(raw)
        quantized.reset(); q_reset = quantized.predict(raw)
        np.testing.assert_array_equal(q_reset, q)
        if len(f) != len(q) or not np.isfinite(f).all():
            raise ValueError("Invalid export scores/lengths")
        delta = np.abs(f-q)
        deltas.extend(delta.tolist())
        fe = event_indices(smooth(f), a.threshold)
        qe = event_indices(smooth(q), a.threshold)
        results.append({"source": row["path"], "engine": row["engine"], "label": row["label"],
                        "steps": len(f), "mean_absolute_error": float(delta.mean()),
                        "max_absolute_error": float(delta.max()),
                        "float_event_steps": fe, "quantized_event_steps": qe,
                        "exact_event_steps_match": fe == qe})
    if not results:
        raise ValueError("No validation speech found")
    report = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "feature_index_sha256": hashlib.sha256((a.features/"features.json").read_bytes()).hexdigest(),
              "tensorflow": tf.__version__, "split": "validation", "threshold": a.threshold,
              "state_buffers": [v.name for v in states], "reset_reproducibility_passed": True,
              "mean_absolute_error": float(np.mean(deltas)),
              "p95_absolute_error": float(np.quantile(deltas,.95)),
              "max_absolute_error": float(np.max(deltas)), "clips": results,
              "limits": "Small deterministic validation subset, not final test. Score drift is expected; full quantized evaluation governs recognition. No MCU numeric equivalence or Hi ESP comparison is established."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
