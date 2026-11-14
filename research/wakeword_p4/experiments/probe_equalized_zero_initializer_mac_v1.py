"""Isolate artifact startup histories on two already consumed fixed witnesses.

Prepare binds the original evidence and makes private model copies changing only
initializer data bytes. Run uses the same original grids, weights, operations,
inputs, clocks and numerical gates. This is not a candidate export or admission.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_report(ref, name):
    path = "research/wakeword_p4/results/" + name
    data = subprocess.check_output(["git", "show", f"{ref}:{path}"])
    return json.loads(data), dict(path=path, sha256=hashlib.sha256(data).hexdigest())


def zero_initializers(data):
    import numpy as np
    from ai_edge_litert import schema_py_generated as s

    original = bytes(data)
    changed = bytearray(data)
    model = s.Model.GetRootAsModel(changed, 0)
    main = model.Subgraphs(0)

    def op_name(op):
        return model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()

    def handles(graph):
        result = {}
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i)
            if op_name(op) == s.BuiltinOperator.VAR_HANDLE:
                options = s.VarHandleOptions()
                raw = op.BuiltinOptions()
                options.Init(raw.Bytes, raw.Pos)
                key = (options.Container(), options.SharedName())
                assert key not in result.values()
                result[int(op.Outputs(0))] = key
        return result

    names = handles(main)
    reads, init_graphs = {}, []
    for i in range(main.OperatorsLength()):
        op = main.Operators(i)
        if op_name(op) == s.BuiltinOperator.READ_VARIABLE:
            reads[names[int(op.Inputs(0))]] = int(op.Outputs(0))
        if op_name(op) == s.BuiltinOperator.CALL_ONCE:
            options = s.CallOnceOptions()
            raw = op.BuiltinOptions()
            options.Init(raw.Bytes, raw.Pos)
            init_graphs.append(options.InitSubgraphIndex())
    assert len(reads) == 6 and len(init_graphs) == 1
    init = model.Subgraphs(init_graphs[0])
    init_handles = handles(init)
    rows, allowed = [], np.zeros(len(data), dtype=bool)
    used = set()
    for i in range(init.OperatorsLength()):
        op = init.Operators(i)
        if op_name(op) != s.BuiltinOperator.ASSIGN_VARIABLE:
            continue
        key = init_handles[int(op.Inputs(0))]
        assert key not in used and key in reads
        used.add(key)
        read = main.Tensors(reads[key])
        tensor = init.Tensors(int(op.Inputs(1)))
        assert tensor.Type() == read.Type() == s.TensorType.INT8
        assert np.array_equal(tensor.ShapeAsNumpy(), read.ShapeAsNumpy())
        q = read.Quantization()
        assert q.ScaleLength() == q.ZeroPointLength() == 1 and q.Scale(0) > 0
        tq = tensor.Quantization()
        assert np.array_equal(tq.ScaleAsNumpy(), q.ScaleAsNumpy())
        assert np.array_equal(tq.ZeroPointAsNumpy(), q.ZeroPointAsNumpy())
        buffer = model.Buffers(tensor.Buffer())
        values = buffer.DataAsNumpy().view(np.int8)
        assert values.size == np.prod(tensor.ShapeAsNumpy())
        # Edit only these exact constant bytes. The flatbuffer is not repacked.
        offset = buffer._tab.Vector(buffer._tab.Offset(4))
        assert bytes(changed[offset:offset + values.size]) == values.tobytes()
        allowed[offset:offset + values.size] = True
        before = values.copy()
        zero = int(q.ZeroPoint(0))
        values[:] = zero
        rows.append(dict(shared_name=key[1].decode(), read=reads[key],
                         shape=tensor.ShapeAsNumpy().tolist(), scale=float(q.Scale(0)),
                         zero_point=zero, buffer=int(tensor.Buffer()), offset=offset,
                         bytes=int(values.size), nonzero_real_codes=int(np.sum(before != zero)),
                         original_sha256=hashlib.sha256(before.tobytes()).hexdigest(),
                         zero_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
    assert used == reads.keys() and len(rows) == 6
    delta = np.frombuffer(original, np.uint8) != np.frombuffer(changed, np.uint8)
    assert not delta[~allowed].any()
    return bytes(changed), rows, int(delta.sum())


def prepare(a):
    ref = subprocess.check_output(["git", "rev-parse", a.gpu_ref], text=True).strip()
    export, export_id = original_report(ref, "4090-hey-broad-equalized-export-result-v1.json")
    screen, screen_id = original_report(ref, "4090-hey-broad-equalized-screen-result-v1.json")
    gates, gates_id = original_report(ref, "4090-hey-broad-trained-quantization-plan-v1.json")
    prior_integer = json.loads((a.integer_replay_root / "report.json").read_text())
    prior_float = json.loads((a.float_replay_root / "report.json").read_text())
    a.output_dir.mkdir(exist_ok=False)
    cases = []
    for target, order in [("hey_willow", 14), ("hey_nyx", 48394)]:
        arm, = [r for r in export["arms"] if r["target"] == target]
        case, = [r for r in screen["cases"] if r["target"] == target and r["request_order"] == order]
        old_i, = [r for r in prior_integer["cases"] if r["target"] == target]
        old_f, = [r for r in prior_float["cases"] if r["target"] == target]
        assert sha(a.payload_root / arm["path"]) == arm["sha256"]
        assert sha(a.payload_root / case["arrays_path"]) == case["arrays_sha256"]
        for root, old in [(a.integer_replay_root, old_i), (a.float_replay_root, old_f)]:
            assert sha(root / old["arrays"]) == old["arrays_sha256"]
        data, rows, changed = zero_initializers((a.payload_root / arm["path"]).read_bytes())
        name = target + "-zero-initializer.tflite"
        (a.output_dir / name).write_bytes(data)
        cases.append(dict(target=target, request_order=order,
                          original_model_path=arm["path"], original_model_sha256=arm["sha256"],
                          source_arrays_path=case["arrays_path"], source_arrays_sha256=case["arrays_sha256"],
                          prior_integer=old_i, prior_float=old_f,
                          zero_model=name, zero_model_sha256=sha(a.output_dir / name),
                          changed_bytes=changed, only_initializer_bytes_changed=True,
                          initializers=rows))
    plan = dict(schema="mac-equalized-zero-initializer-plan-v1", gpu_ref=ref,
                source_reports=[export_id, screen_id, gates_id],
                script_sha256=sha(__file__), numerical_gates=gates["numerical_gates"], cases=cases,
                prior_integer_report_sha256=sha(a.integer_replay_root / "report.json"),
                prior_float_report_sha256=sha(a.float_replay_root / "report.json"),
                policy="Two fixed consumed training witnesses only. Private copies change only six initializer constant buffers to their existing read-grid zero-points; no weight/grid/op/shape/clock/gate change, export, optimizer, waveform IO, new development/acceptance or board write. Original fresh-artifact results retained. Repeated fresh zero-initializer interpreters compare full streams and all observed integer tensors with original replay; actual Mac explicit-zero float remains unchanged. No candidate admission from these diagnostics.")
    with a.plan.open("x") as f:
        f.write(json.dumps(plan, indent=2) + "\n")


def run(a):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    plan = json.loads(a.plan.read_text())
    assert sha(__file__) == plan["script_sha256"]
    assert sha(a.integer_replay_root / "report.json") == plan["prior_integer_report_sha256"]
    assert sha(a.float_replay_root / "report.json") == plan["prior_float_report_sha256"]
    versions = {n: importlib.metadata.version(n) for n in ["numpy", "ai-edge-litert"]}
    assert versions == {"numpy": "2.4.6", "ai-edge-litert": "2.3.0"}
    gates = plan["numerical_gates"]

    def stats(error):
        five = np.abs(np.convolve(np.asarray(error, np.float64), np.ones(5) / 5, "valid"))
        mean, p99, maximum = float(five.mean()), float(np.percentile(five, 99)), float(five.max())
        passed = (mean <= gates["per_input_five_step_mean_abs_error_max"]
                  and p99 <= gates["per_input_five_step_p99_abs_error_max"]
                  and maximum <= gates["per_input_five_step_max_abs_error_max"])
        return dict(mean=mean, p99=p99, maximum=maximum, passed=bool(passed))

    results = []
    for c in plan["cases"]:
        assert sha(a.payload_root / c["original_model_path"]) == c["original_model_sha256"]
        assert sha(a.payload_root / c["source_arrays_path"]) == c["source_arrays_sha256"]
        assert sha(a.output_dir / c["zero_model"]) == c["zero_model_sha256"]
        assert sha(a.integer_replay_root / c["prior_integer"]["arrays"]) == c["prior_integer"]["arrays_sha256"]
        assert sha(a.float_replay_root / c["prior_float"]["arrays"]) == c["prior_float"]["arrays_sha256"]
        with np.load(a.payload_root / c["source_arrays_path"], allow_pickle=False) as saved:
            q, old_codes = saved["quantized_input"].copy(), saved["uint8"].copy()
        with np.load(a.integer_replay_root / c["prior_integer"]["arrays"] , allow_pickle=False) as saved:
            old_tensors = {int(k[7:]): saved[k].copy() for k in saved.files if k.startswith("tensor_")}
        with np.load(a.float_replay_root / c["prior_float"]["arrays"], allow_pickle=False) as saved:
            baseline = saved["probability"].copy()
        assert len(q) // 3 == len(baseline) == len(old_codes)
        traces = []
        for _ in range(2):
            interpreter = Interpreter(model_path=str(a.output_dir / c["zero_model"]), num_threads=1,
                                      experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                                      experimental_preserve_all_tensors=True)
            interpreter.allocate_tensors()
            input_id = interpreter.get_input_details()[0]["index"]
            output = interpreter.get_output_details()[0]
            rows = {i: [] for i in old_tensors}
            for at in range(0, len(q), 3):
                interpreter.set_tensor(input_id, q[None, at:at + 3])
                interpreter.invoke()
                for i in rows:
                    rows[i].append(interpreter.get_tensor(i).copy())
            trace = {i: np.stack(v) for i, v in rows.items()}
            for state in c["initializers"]:
                assert np.all(trace[state["read"]][0] == state["zero_point"])
            traces.append(trace)
        assert all(np.array_equal(traces[0][i], traces[1][i]) for i in old_tensors)
        trace = traces[0]
        codes = trace[output["index"]].reshape(-1)
        scale, zero = output["quantization"]
        original = (old_codes.astype(np.float64) - zero) * scale
        controlled = (codes.astype(np.float64) - zero) * scale
        tensor_differences = {}
        for i, old in old_tensors.items():
            unequal = np.any(trace[i] != old, axis=tuple(range(1, old.ndim)))
            indices = np.flatnonzero(unequal)
            tensor_differences[str(i)] = dict(unequal_steps=int(len(indices)),
                                            last_unequal_step_zero_based=int(indices[-1]) if len(indices) else None)
        all_last = [v["last_unequal_step_zero_based"] for v in tensor_differences.values()
                    if v["last_unequal_step_zero_based"] is not None]
        last = max(all_last) if all_last else None
        filename = c["target"] + "-zero-replay.npz"
        with (a.output_dir / filename).open("xb") as f:
            np.savez_compressed(f, probability=controlled, float_reference=baseline,
                                original_probability=original, **{f"tensor_{i}": v for i, v in trace.items()})
        result = dict(target=c["target"], request_order=c["request_order"], steps=len(codes),
                      original_error=stats(original - baseline), zero_initializer_error=stats(controlled - baseline),
                      original_vs_zero_error=stats(controlled - original), twice_fresh_exact=True,
                      all_six_first_read_states_are_real_zero=True,
                      all_observed_tensors_equal_original_after_step_zero_based=last,
                      tensor_differences=tensor_differences, arrays=filename,
                      arrays_sha256=sha(a.output_dir / filename))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(schema="mac-equalized-zero-initializer-result-v1", plan_sha256=sha(a.plan),
                  script_sha256=sha(__file__), versions=versions, cases=results,
                  candidate_admitted=False, limits=plan["policy"])
    with (a.output_dir / "report.json").open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "run"])
    parser.add_argument("--gpu-ref", default="99e146a")
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--integer-replay-root", type=Path, required=True)
    parser.add_argument("--float-replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.phase == "prepare" else run(args)
