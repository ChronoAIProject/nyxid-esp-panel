"""Stress the pinned GPU STE primitive with independent integer arithmetic.

These constructed accumulator controls are not learned-model inference,
recognition evidence, or permission to begin formal QAT.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def factor(value):
    # Independent rational half-up rounding of the positive frexp significand.
    f, shift = math.frexp(float(value))
    numerator, denominator = f.as_integer_ratio()
    q, remainder = divmod(numerator * (1 << 31), denominator)
    q += int(2 * remainder >= denominator)
    if q == 1 << 31:
        q //= 2
        shift += 1
    return (0, 0) if shift < -31 else (q, shift)


def integer_code(value, multiplier, zero, low, high):
    q, shift = factor(multiplier)
    left_value = int(value) * (1 << max(shift, 0))
    assert -(1 << 31) <= left_value < 1 << 31
    # High multiply rounds ties toward +infinity. The following power-of-two
    # divide instead rounds signed ties away from zero, as in gemmlowp.
    high_value = (left_value * q + (1 << 30)) // (1 << 31)
    right = max(-shift, 0)
    if right:
        rounded = (abs(high_value) + (1 << (right - 1))) // (1 << right)
        high_value = rounded if high_value >= 0 else -rounded
    return min(high, max(low, high_value + zero))


def points(multiplier, zero, low, high):
    q, shift = factor(multiplier)
    left = 1 << max(shift, 0)
    lower, upper = -(1 << 31) // left, ((1 << 31) - 1) // left
    values = {lower, lower + 1, upper - 1, upper, -1, 0, 1}
    for n in [2, 3, 127, 128, 255, 256, (1 << 24) - 1, 1 << 24, (1 << 24) + 1]:
        values.update([n, -n])
    if multiplier:
        for boundary in [low - zero, high - zero, low - zero - .5, high - zero + .5]:
            at = math.floor(boundary / multiplier)
            values.update(range(at - 2, at + 4))
    # Solve exact high-multiply half-ties modulo 2^31, then include neighbours.
    if q and shift <= 0:
        divisor = math.gcd(q, 1 << 31)
        if (1 << 30) % divisor == 0:
            modulus = (1 << 31) // divisor
            tie = ((1 << 30) // divisor * pow(q // divisor, -1, modulus)) % modulus
            for at in [tie, tie - modulus]:
                values.update([at - 1, at, at + 1])
    result = sorted(v for v in values if lower <= v <= upper)
    assert len(result) <= 64
    return result + [0] * (64 - len(result))


def review(a):
    import numpy as np
    import tensorflow as tf
    from ai_edge_litert import schema_py_generated as schema

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    a.output_dir.mkdir(exist_ok=False)
    source_dir = a.output_dir / "source"
    source_dir.mkdir()
    ref = subprocess.check_output(["git", "rev-parse", a.gpu_ref], text=True).strip()
    sources = {}
    for name in ["hey_broad_fixedpoint_ste_v1.py", "probe_hey_broad_integer_requant_v2.py"]:
        path = "research/wakeword_p4/experiments/" + name
        data = subprocess.check_output(["git", "show", f"{ref}:{path}"])
        (source_dir / name).write_bytes(data)
        sources[path] = hashlib.sha256(data).hexdigest()
    sys.path.insert(0, str(source_dir))
    from hey_broad_fixedpoint_ste_v1 import requantize_ste

    plan = json.loads(a.initializer_plan.read_text())
    groups = []
    models = {}
    for arm in plan["cases"]:
        path = a.payload_root / arm["original_model_path"]
        assert sha(path) == arm["original_model_sha256"]
        models[arm["target"]] = sha(path)
        model = schema.Model.GetRootAsModel(path.read_bytes(), 0)
        graph = model.Subgraphs(0)
        for j in range(graph.OperatorsLength()):
            op = graph.Operators(j)
            kind = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
            if kind not in [schema.BuiltinOperator.CONV_2D, schema.BuiltinOperator.DEPTHWISE_CONV_2D]:
                continue
            inputs = [graph.Tensors(int(v)) for v in op.InputsAsNumpy()]
            output = graph.Tensors(int(op.Outputs(0)))
            iq, wq, oq = inputs[0].Quantization(), inputs[1].Quantization(), output.Quantization()
            multipliers = float(iq.Scale(0)) * wq.ScaleAsNumpy().astype(np.float64) / float(oq.Scale(0))
            low = max(-128, int(oq.ZeroPoint(0))) if kind == schema.BuiltinOperator.CONV_2D else -128
            groups.append((arm["target"] + "/" + str(int(op.Outputs(0))), multipliers,
                           int(oq.ZeroPoint(0)), low, 127, float(oq.Scale(0))))
    assert len(groups) == 18
    for name, multipliers, zero, low, high, scale in [
        ("signed-ties", [0, .5, .25, .75, math.nextafter(.5, 0), math.nextafter(.5, 1)], -7, -128, 127, 1.),
        ("shifts-underflow", [2. ** -40, 2. ** -32, 2. ** -31, 2. ** -16, 1., 2., 4.], 0, -128, 127, .25),
        ("relu-saturation", [.5, .25, .125, .75], -128, -128, 127, 1. / 256),
    ]:
        groups.append((name, np.asarray(multipliers, np.float64), zero, low, high, scale))
    reports = []
    for name, multipliers, zero, low, high, scale in groups:
        values = np.asarray([points(v, zero, low, high) for v in multipliers], np.float64).T
        expected_codes = np.asarray([[integer_code(x, m, zero, low, high)
                                      for x, m in zip(row, multipliers, strict=True)] for row in values], np.int64)
        expected = (expected_codes - zero) * scale
        x = tf.Variable(values, dtype=tf.float64)
        weight = ((np.arange(values.size).reshape(values.shape) % 13) - 6) / 8.
        before = x.numpy().copy()
        with tf.GradientTape() as tape:
            actual = requantize_ste(x, multipliers, zero, low, high, scale, tf)
            loss = tf.reduce_sum(actual * tf.constant(weight, tf.float64))
        gradient = tape.gradient(loss, x).numpy()
        continuous = values * (multipliers * scale) + zero * scale
        active = (continuous >= low * scale) & (continuous <= high * scale)
        expected_gradient = weight * (multipliers * scale) * active
        assert np.array_equal(actual.numpy(), expected), name
        assert np.array_equal(gradient, expected_gradient), name
        assert np.array_equal(x.numpy(), before)
        assert np.isfinite(gradient).all()
        reports.append(dict(name=name, multipliers=multipliers.tolist(), zero=zero, low=low, high=high,
                            output_scale=scale, points=int(values.size), forward_exact=True,
                            declared_analytic_gradient_exact=True, variable_unchanged=True))
    result = dict(schema="mac-fixedpoint-ste-boundary-review-v1", gpu_ref=ref,
                  script_sha256=sha(__file__), source_sha256=sources, models=models,
                  initializer_plan_sha256=sha(a.initializer_plan), groups=reports,
                  total_points=sum(g["points"] for g in reports),
                  versions={n: importlib.metadata.version(n) for n in ["tensorflow", "numpy"]},
                  model_invocations=0, optimizer_updates=0,
                  limits="Constructed 64 accumulator points per actual encoder channel plus explicit signed tie, clipping, underflow and positive-shift groups. Python exact-integer reference is separate from pinned GPU multiplier implementation. Left-shifted accumulators are restricted to int32 representable domain. Forward and declared continuous clipped-STE gradient are checked, not a derivative of integer steps. Not fullgraph/actual-batch gradient parity, native runtime/P4 kernel equivalence, QAT success, recognition or candidate admission.")
    with (a.output_dir / "report.json").open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(groups=len(reports), total_points=result["total_points"], passed=True)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ref", required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--initializer-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    review(parser.parse_args())
