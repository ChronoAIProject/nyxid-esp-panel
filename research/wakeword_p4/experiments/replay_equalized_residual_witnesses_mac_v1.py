"""Independently replay the two fixed residual failures and split head errors.

Only the six previously published model/weight/NPZ artifacts are read. The
bounded float baseline is saved GPU evidence; integer inference and the hybrid
boundary calculations execute here. This does not admit a candidate or select
an operating point.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from infer_quantized_state_grids_v1 import state_grid_layout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(ref, payload, out):
    import numpy as np
    import h5py
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    ref = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    reports = {}
    def read(name):
        path = "research/wakeword_p4/results/" + name
        data = subprocess.check_output(["git", "show", f"{ref}:{path}"])
        reports[path] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    export = read("4090-hey-broad-equalized-export-result-v1.json")
    receipt = read("4090-hey-broad-equalized-weights-receipt-v1.json")
    screen = read("4090-hey-broad-equalized-screen-result-v1.json")
    plan = read("4090-hey-broad-trained-quantization-plan-v1.json")
    gates = plan["numerical_gates"]
    targets = {"hey_willow": 14, "hey_nyx": 48394}
    artifacts = {}
    selected = []
    for target, order in targets.items():
        arm, = [r for r in export["arms"] if r["target"] == target]
        weights, = [r for r in receipt["arms"] if r["target"] == target]
        scored, = [r for r in screen["cases"] if r["target"] == target and r["request_order"] == order]
        request, = [r for r in plan["requests"] if r["order"] == order]
        for row, key in [(arm, "path"), (weights, "path"), (scored, "arrays_path")]:
            expected = row["arrays_sha256"] if key == "arrays_path" else row["sha256"]
            assert sha(payload / row[key]) == expected, row[key]
            artifacts[row[key]] = expected
        selected.append((arm, weights, scored, request))
    assert len(artifacts) == 6
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "h5py", "ai-edge-litert")}
    assert versions["numpy"] == "2.4.6" and versions["ai-edge-litert"] == "2.3.0"
    out.mkdir(exist_ok=False)

    def sigmoid(value):
        return np.exp(-np.logaddexp(0, -np.asarray(value, np.float64)))

    def stats(delta):
        delta = np.asarray(delta, np.float64)
        five = np.asarray([np.sum(delta[i:i+5]) / 5 for i in range(len(delta)-4)])
        absolute = np.abs(five)
        return dict(raw_max_abs=float(np.max(np.abs(delta))),
                    five_mean_abs=float(np.mean(absolute)), five_p99_abs=float(np.percentile(absolute, 99)),
                    five_max_abs=float(np.max(absolute)))

    def passed(m):
        return bool(m["five_mean_abs"] <= gates["per_input_five_step_mean_abs_error_max"]
                    and m["five_p99_abs"] <= gates["per_input_five_step_p99_abs_error_max"]
                    and m["five_max_abs"] <= gates["per_input_five_step_max_abs_error_max"])

    def dequant(value, detail):
        p = detail["quantization_parameters"]
        scale, zero = p["scales"].astype(np.float64), p["zero_points"].astype(np.float64)
        assert len(scale) and (scale > 0).all()
        if len(scale) > 1:
            shape = [1] * value.ndim
            shape[p["quantized_dimension"]] = len(scale)
            scale, zero = scale.reshape(shape), zero.reshape(shape)
        return (value.astype(np.float64) - zero) * scale

    cases = []
    for arm, weights, scored, request in selected:
        with np.load(payload / scored["arrays_path"], allow_pickle=False) as saved:
            q, raw, A, baseline = [saved[k].copy() for k in ("quantized_input", "raw", "float", "uint8")]
            assert np.array_equal(baseline, saved["uint8_repeat"])
            assert np.array_equal(A, saved["float_repeat"])
            clocks = (3*np.arange(1, len(A)+1)+2)*160
            assert np.array_equal(clocks, saved["support_unclamped"])
            assert np.array_equal(np.minimum(clocks, request["samples"]), saved["sample_end"])
        assert len(q) % 3 == 0 and len(A) == len(q)//3
        repetitions = []
        metadata = None
        for _ in range(2):
            interpreter = Interpreter(model_path=str(payload / arm["path"]), num_threads=1,
                                      experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                                      experimental_preserve_all_tensors=True)
            interpreter.allocate_tensors()
            details = {d["index"]: d for d in interpreter.get_tensor_details()}
            inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            assert inp["dtype"] == np.int8 and output["dtype"] == np.uint8
            assert list(inp["shape"]) == [1, 3, 40] and list(output["shape"]) == [1, 1]
            values = np.clip(np.pad(raw, ((0, (-len(raw)) % 3), (0, 0))).astype(np.float32)/25.6, 0, 26)
            scale, zero = inp["quantization"]
            assert np.array_equal(q, np.clip(np.rint(values/scale)+zero, -128, 127).astype(np.int8))
            ids = sorted({i for op in arm["ops"] for i in op["outputs"]
                          if details[i]["dtype"] in (np.int8, np.uint8) and details[i]["quantization"][0] > 0})
            trace = {i: [] for i in ids}
            for at in range(0, len(q), 3):
                interpreter.set_tensor(inp["index"], q[None, at:at+3])
                interpreter.invoke()
                for i in ids:
                    trace[i].append(interpreter.get_tensor(i).copy())
            trace = {i: np.stack(rows) for i, rows in trace.items()}
            assert np.array_equal(trace[output["index"]].reshape(-1), baseline)
            repetitions.append(trace)
            metadata = details
        assert all(np.array_equal(repetitions[0][i], repetitions[1][i]) for i in ids)
        trace = repetitions[0]
        def grid(index):
            d = metadata[index]
            return dict(scales=d["quantization_parameters"]["scales"].tolist(),
                        zeros=d["quantization_parameters"]["zero_points"].tolist(),
                        shape=d["shape"].tolist(), dtype=str(d["dtype"]))
        layout = state_grid_layout(arm["ops"], grid)
        state_checks = []
        for binding in layout["state_bindings"]:
            read, assigned, current, concat = [trace[binding[k]] for k in ("read", "assigned", "current", "concatenation")]
            state_checks.append(dict(handle=binding["handle"],
                                     next_read_equals_previous_assignment=bool(np.array_equal(read[1:], assigned[:-1])),
                                     concat_equals_history_then_current=bool(np.array_equal(concat, np.concatenate((read, current), axis=2))),
                                     state_transition_pairs=len(A)-1))
        fc, = [op for op in arm["ops"] if op["op"] == "FULLY_CONNECTED"]
        embedding_index, weight_index, bias_index = fc["inputs"]
        fc_index, = fc["outputs"]
        embedding = dequant(trace[embedding_index], metadata[embedding_index]).reshape(len(A), -1)
        with h5py.File(payload / weights["path"]) as h:
            w, b = np.array(h["layers/dense/vars/0"]), np.array(h["layers/dense/vars/1"])
        assert w.shape == (embedding.shape[1], 1) and b.shape == (1,)
        B = sigmoid(np.clip((embedding @ w.astype(np.float64)+b).reshape(-1), -8, 8))
        qw = dequant(interpreter.get_tensor(weight_index), metadata[weight_index])
        qb = dequant(interpreter.get_tensor(bias_index), metadata[bias_index])
        D = sigmoid(np.clip((embedding @ qw.T+qb).reshape(-1), 0, 6)*(8/3)-8)
        E = sigmoid(dequant(trace[fc_index], metadata[fc_index]).reshape(-1)*(8/3)-8)
        F = dequant(trace[output["index"]], metadata[output["index"]]).reshape(-1)
        boundaries = dict(encoder_embedding=B-A, head_weight_bias=D-B, integer_fc=E-D, post_fc_probability=F-E)
        assert np.max(np.abs(sum(boundaries.values())-(F-A))) <= 1e-12
        total = stats(F-A)
        for key, name in [("mean", "five_mean_abs"), ("p99", "five_p99_abs"), ("maximum", "five_max_abs")]:
            assert abs(total[name]-scored["smoothed_abs_error"][key]) <= 1e-12
        assert passed(total) == scored["passed"]
        filename = arm["target"]+".npz"
        np.savez_compressed(out/filename, A=A, B=B, D=D, E=E, F=F,
                            **{f"tensor_{i}": trace[i] for i in ids})
        result = dict(target=arm["target"], request_order=scored["request_order"], steps=len(A),
                      original_uint8_byte_exact=True, all_tensor_repeats_exact=True, observed_tensors=len(ids),
                      boundaries={k: stats(v) for k, v in boundaries.items()},
                      hybrid_original_float_head=stats(B-A), hybrid_passed_on_this_witness=passed(stats(B-A)),
                      actual_total=total, original_gate_passed=passed(total), state_checks=state_checks,
                      arrays=filename, arrays_sha256=sha(out/filename))
        cases.append(result)
        print(json.dumps(result), flush=True)
    result = dict(schema="mac-equalized-residual-replay-v1", gpu_ref=ref, source_reports=reports,
                  artifacts=artifacts, versions=versions, script_sha256=sha(__file__), cases=cases,
                  limits="Two fixed selected training failures only. GPU saved float A is not independently recomputed in this run. Actual Mac builtin-ref integer outputs, all tensor reset repeats, and hybrid head calculations executed here. Boundary maxima occur at different steps and cannot be added as causal shares. A hybrid pass on a witness does not establish all270, recognition/FAR, new development, P4 performance or candidate admission. No optimizer, export, waveform playback/recording or board write.")
    with (out/"report.json").open("x") as f:
        f.write(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-ref", required=True)
    p.add_argument("--payload-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    replay(a.gpu_ref, a.payload_root, a.output_dir)
