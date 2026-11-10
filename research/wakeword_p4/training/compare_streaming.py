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
    p.add_argument("--engine", action="append")
    p.add_argument("--label", type=int, choices=(0, 1), action="append")
    p.add_argument("--variants", type=int, nargs="+", default=[0])
    p.add_argument("--upstream", type=Path, help="Also isolate non-stream window phase alignment")
    p.add_argument("--float-reset-policy", choices=("zeros", "exported"), default="zeros",
                   help="Compare explicit zero buffers or the initial states saved in the export")
    a = p.parse_args()
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    loaded = tf.saved_model.load(str(a.model_dir/"saved_model"))
    window_model = None
    report = json.loads((a.model_dir/"training.json").read_text())
    frames = report.get("context_frames", 180)
    phase = report.get("validation_window_phase", 1)
    if a.upstream:
        import sys
        sys.path.insert(0, str(a.upstream.resolve()))
        from microwakeword import mixednet
        report = json.loads((a.model_dir/"training.json").read_text())
        arch = argparse.ArgumentParser()
        mixednet.model_parameters(arch)
        window_model = mixednet.model(arch.parse_args(report["architecture_args"]), (frames,40), None)
        window_model.load_weights(a.model_dir/"best.weights.h5")
    signature = loaded.signatures["serving_default"]
    keys = list(signature.structured_input_signature[1])
    if len(keys) != 1:
        raise ValueError("Expected one streaming feature input")
    states = [v for v in loaded.variables if "states" in v.name]
    if not states:
        raise ValueError("No float stream buffers found; cannot verify reset")
    exported_states = [v.numpy().copy() for v in states]
    def predict_float(raw):
        for state, initial in zip(states, exported_states):
            state.assign(initial if a.float_reset_policy == "exported" else tf.zeros_like(state))
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
    mature_deltas = []
    # First step whose complete window, including the trained phase, is present.
    # This is a diagnostic partition only: no samples/events are discarded.
    complete_context_step = (frames + phase + 2)//3 - 1
    for row in rows:
        if row["split"] != "validation" or row["variant"] not in a.variants:
            continue
        if a.engine and row["engine"] not in a.engine:
            continue
        if a.label and row["label"] not in a.label:
            continue
        key = (row["engine"], row["label"])
        if a.per_group and groups[key] >= a.per_group:
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
        mature = delta[complete_context_step:]
        mature_deltas.extend(mature.tolist())
        fe = event_indices(smooth(f), a.threshold)
        qe = event_indices(smooth(q), a.threshold)
        end = (row["keyword_frame"]+2)*.01 if row["label"] else None
        def hit(indices):
            return any(end-.1 <= (3*(i+4)+5)*.01 <= end+.8 for i in indices) if end is not None else None
        context = None
        if window_model is not None:
            x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
            ends = [3*(i+1) for i in range(len(f)) if frames <= 3*(i+1) and 3*(i+1)+phase <= len(x)]
            baseline = np.stack([x[e-frames:e] for e in ends])
            aligned = np.stack([x[e-frames+phase:e+phase] for e in ends])
            b = window_model(baseline, training=False).numpy().reshape(-1)
            c = window_model(aligned, training=False).numpy().reshape(-1)
            reference = f[np.asarray(ends)//3-1]
            context = {"original_phase_max_error": float(np.max(np.abs(b-reference))),
                       "aligned_phase_max_error": float(np.max(np.abs(c-reference))),
                       "aligned_phase_mean_error": float(np.mean(np.abs(c-reference)))}
        results.append({"source": row["path"], "engine": row["engine"], "label": row["label"],
                        "variant": row["variant"], "float_keyword_hit": hit(fe), "quantized_keyword_hit": hit(qe),
                        "window_context": context,
                        "steps": len(f), "mean_absolute_error": float(delta.mean()),
                        "max_absolute_error": float(delta.max()),
                        "drift_peak_step": int(np.argmax(delta)),
                        "drift_peak_audio_seconds": float((3*np.argmax(delta)+5)*.01),
                        "initial_context_max_absolute_error": float(delta[:complete_context_step].max()) if complete_context_step else None,
                        "complete_context_mean_absolute_error": float(mature.mean()) if len(mature) else None,
                        "complete_context_max_absolute_error": float(mature.max()) if len(mature) else None,
                        "initial_float_scores": f[:complete_context_step].tolist(),
                        "initial_quantized_scores": q[:complete_context_step].tolist(),
                        "float_event_steps": fe, "quantized_event_steps": qe,
                        "exact_event_steps_match": fe == qe})
    if not results:
        raise ValueError("No validation speech found")
    report = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "feature_index_sha256": hashlib.sha256((a.features/"features.json").read_bytes()).hexdigest(),
              "tensorflow": tf.__version__, "split": "validation", "threshold": a.threshold,
              "float_reset_policy": a.float_reset_policy,
              "exported_state_initial_values": [dict(name=v.name, nonzero=int(np.count_nonzero(x)),
                                                     sha256=hashlib.sha256(x.tobytes()).hexdigest())
                                                for v, x in zip(states, exported_states)],
              "state_buffers": [v.name for v in states], "reset_reproducibility_passed": True,
              "mean_absolute_error": float(np.mean(deltas)),
              "p95_absolute_error": float(np.quantile(deltas,.95)),
              "max_absolute_error": float(np.max(deltas)), "clips": results,
              "complete_context_first_step": complete_context_step,
              "complete_context_mean_absolute_error": float(np.mean(mature_deltas)) if mature_deltas else None,
              "complete_context_max_absolute_error": float(np.max(mature_deltas)) if mature_deltas else None,
              "all_initial_scores_and_events_retained": True,
              "positive_clips": sum(r["label"] == 1 for r in results),
              "float_keyword_hits": sum(r["float_keyword_hit"] is True for r in results),
              "quantized_keyword_hits": sum(r["quantized_keyword_hit"] is True for r in results),
              "limits": "Small deterministic validation subset, not final test. Score drift is expected; full quantized evaluation governs recognition. No MCU numeric equivalence or Hi ESP comparison is established."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
