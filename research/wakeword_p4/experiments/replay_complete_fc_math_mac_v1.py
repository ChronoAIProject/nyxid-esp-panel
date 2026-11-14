"""Execute the complete-runtime FC header on the original saved witnesses.

The oracle for model cases is the original actual LiteRT trace, not a Python
port of this header. Synthetic boundary controls use exact rational arithmetic.
This does not execute TFLM preparation or a P4 interpreter.
"""
import argparse
import ctypes
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def round_fraction(value):
    """Exact signed nearest rounding with half ties away from zero."""
    negative = value < 0
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude.numerator, magnitude.denominator)
    rounded = quotient + (2 * remainder >= magnitude.denominator)
    return -rounded if negative else rounded


def replay(a):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    repo = Path(__file__).resolve().parents[3]
    a.output_dir.mkdir(exist_ok=False)
    runtime = repo / 'research/panel_wake_runtime'
    wrapper = runtime / 'complete_fc_math_host_v1.cc'
    library = a.output_dir / 'libcomplete_fc_math.dylib'
    compiler = shutil.which('clang++')
    command = [compiler, '-std=c++17', '-O2', '-dynamiclib', '-Wall', '-Wextra',
               '-Werror', '-I' + str(runtime), str(wrapper), '-o', str(library)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (a.output_dir / 'compile.json').write_text(json.dumps(vars(compiled), indent=2) + '\n')
    compiled.check_returncode()
    dylib = ctypes.CDLL(str(library))
    fc = dylib.complete_fc_math_host_v1
    p8, p32 = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)
    fc.argtypes = [p8, p8, p32, p8] + [ctypes.c_int] * 4 + [ctypes.c_double] + [ctypes.c_int] * 2
    fc.restype = None

    def execute(inputs, weights, bias, input_zero, output_zero, scale, low, high):
        inputs = np.ascontiguousarray(inputs, dtype=np.int8)
        weights = np.ascontiguousarray(weights, dtype=np.int8)
        assert inputs.ndim == 2 and weights.shape == (inputs.shape[1],)
        assert inputs.shape[0] > 0 and 0 < inputs.shape[1] <= 4096
        assert -128 <= low <= high <= 127 and scale > 0
        b = None if bias is None else np.asarray([bias], np.int32)
        output = np.empty(inputs.shape[0], np.int8)
        fc(inputs.ctypes.data_as(p8), weights.ctypes.data_as(p8),
           None if b is None else b.ctypes.data_as(p32), output.ctypes.data_as(p8),
           *inputs.shape, input_zero, output_zero, scale, low, high)
        return output

    # Power-of-two rational scales keep the exact oracle independent of any
    # host floating rounding routine. Exercise signed ties, clipping, nonzero
    # zero-points, absent/present biases, extreme legal accumulators and batches.
    controls = []
    control_codes = 0
    seed = np.random.default_rng(20261114)
    groups = []
    for bias in [None, -3, 5]:
        for scale in [Fraction(1, 2), Fraction(3, 2), Fraction(1, 256)]:
            for low, high in [(-128, 127), (-37, 43)]:
                groups.append((np.arange(-128, 128, dtype=np.int16).astype(np.int8)[:, None],
                               np.asarray([1], np.int8), bias, 0, 7, scale, low, high))
    for depth in [7, 1800, 4096]:
        for zero in [-128, 0, 127]:
            x = seed.integers(-128, 128, (19, depth), dtype=np.int8)
            w = seed.integers(-128, 128, depth, dtype=np.int8)
            groups.append((x, w, -71, zero, -13, Fraction(1, 256), -128, 127))
    for bias in [-(2**31) + 40000, (2**31) - 40000]:
        groups.append((np.asarray([[-128], [127]], np.int8), np.asarray([-128], np.int8),
                       bias, 0, 0, Fraction(1, 2**24), -128, 127))
    for number, (x, w, bias, iz, oz, scale, low, high) in enumerate(groups):
        expected = []
        accumulators = []
        for row in x:
            acc = (bias or 0) + sum(int(weight) * (int(v) - iz) for weight, v in zip(w, row))
            assert -(2**31) <= acc < 2**31
            accumulators.append(acc)
            expected.append(max(low, min(high, round_fraction(acc * scale) + oz)))
        first = execute(x, w, bias, iz, oz, float(scale), low, high)
        second = execute(x, w, bias, iz, oz, float(scale), low, high)
        exact = np.array_equal(first, np.asarray(expected, np.int8)) and np.array_equal(first, second)
        controls.append(dict(group=number, batches=len(x), depth=x.shape[1],
                             input_zero=iz, output_zero=oz, bias=bias, scale=str(scale),
                             low=low, high=high, accumulator_min=min(accumulators),
                             accumulator_max=max(accumulators), exact=bool(exact)))
        control_codes += len(x)

    ref = subprocess.check_output(['git', 'rev-parse', a.gpu_ref], text=True).strip()
    raw = subprocess.check_output(['git', 'show', ref + ':research/wakeword_p4/results/4090-hey-broad-equalized-export-result-v1.json'])
    export = json.loads(raw)
    prior = json.loads((a.integer_replay_root / 'report.json').read_text())
    ansi = json.loads((a.ansi_replay_root / 'report.json').read_text())
    results = []
    for target in ['hey_willow', 'hey_nyx']:
        arm = next(v for v in export['arms'] if v['target'] == target)
        case = next(v for v in prior['cases'] if v['target'] == target)
        ansi_case = next(v for v in ansi['cases'] if v['target'] == target)
        path = a.payload_root / arm['path']
        assert sha(path) == arm['sha256']
        assert sha(a.integer_replay_root / case['arrays']) == case['arrays_sha256']
        assert sha(a.ansi_replay_root / ansi_case['arrays']) == ansi_case['arrays_sha256']
        original = Interpreter(model_content=path.read_bytes(),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        original.allocate_tensors()
        details = {d['index']: d for d in original.get_tensor_details()}
        op, = [o for o in arm['ops'] if o['op'] == 'FULLY_CONNECTED']
        embedding_id, weight_id, bias_id = op['inputs']
        output_id, = op['outputs']
        assert op['fused_activation'] == 3  # actual model RELU6
        weights = original.get_tensor(weight_id).copy()
        bias, = original.get_tensor(bias_id).reshape(-1)
        iscale, izero = details[embedding_id]['quantization']
        wscale, wzero = details[weight_id]['quantization']
        oscale, ozero = details[output_id]['quantization']
        assert wzero == 0 and weights.shape == (1, 1800)
        # Python float widens the actual float32 metadata before multiplication.
        effective = float(iscale) * float(wscale) / float(oscale)
        low = max(-128, ozero)
        high = min(127, ozero + round_fraction(Fraction(6) / Fraction(oscale)))
        with np.load(a.integer_replay_root / case['arrays'], allow_pickle=False) as saved:
            embeddings = saved[f'tensor_{embedding_id}'].copy().reshape(-1, 1800)
            host_fc = saved[f'tensor_{output_id}'].copy().reshape(-1)
            host_final = saved['F'].copy()
        with np.load(a.ansi_replay_root / ansi_case['arrays'], allow_pickle=False) as saved:
            old_fc = saved['p4_ansi_fc'].copy()
            assert np.array_equal(saved['host_fc'], host_fc)
            assert np.array_equal(saved['original_uint8'].astype(np.float64) / 256, host_final)
        values = execute(embeddings, weights[0], bias, izero, ozero, effective, low, high)
        repeat = execute(embeddings, weights[0], bias, izero, ozero, effective, low, high)
        individual = np.concatenate([execute(row[None, :], weights[0], bias, izero, ozero,
                                            effective, low, high) for row in embeddings])

        fragment = a.ansi_replay_root / (target + '-post-fc.tflite')
        assert sha(fragment) == ansi_case['fragment_sha256']
        post = Interpreter(model_content=fragment.read_bytes(),
            experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        post.allocate_tensors()
        inp, out = post.get_input_details()[0], post.get_output_details()[0]
        def downstream(codes):
            result = []
            for code in codes:
                post.set_tensor(inp['index'], np.asarray([[code]], np.int8))
                post.invoke()
                result.append(int(post.get_tensor(out['index'])[0, 0]))
            return np.asarray(result, np.uint8)
        control = downstream(host_fc)
        actual = downstream(values)
        fc_exact = np.array_equal(values, host_fc)
        downstream_control_exact = np.array_equal(control.astype(np.float64) / 256, host_final)
        downstream_exact = np.array_equal(actual, control)
        batch_exact = np.array_equal(values, individual)
        repeat_exact = np.array_equal(values, repeat)
        arrays = a.output_dir / (target + '-fc-replay.npz')
        np.savez_compressed(arrays, original_fc=host_fc, complete_fc=values,
                            complete_fc_repeat=repeat, complete_fc_individual=individual,
                            original_uint8=control, complete_fc_post_uint8=actual)
        result = dict(target=target, steps=len(embeddings), model_sha256=sha(path),
            effective_scale=effective, input_zero=izero, output_zero=ozero,
            activation_min=low, activation_max=high, original_ansi_unequal_steps=int(np.count_nonzero(old_fc != host_fc)),
            fc_unequal_steps=int(np.count_nonzero(values != host_fc)),
            final_uint8_unequal_steps=int(np.count_nonzero(actual != control)),
            fc_exact=bool(fc_exact), downstream_control_exact=bool(downstream_control_exact),
            downstream_exact=bool(downstream_exact), batch_exact=bool(batch_exact), repeat_exact=bool(repeat_exact),
            arrays=arrays.name, arrays_sha256=sha(arrays))
        results.append(result)
        print(json.dumps(result), flush=True)
    report = dict(schema='mac-complete-fc-math-replay-v1', gpu_ref=ref,
        script_sha256=sha(__file__), wrapper_sha256=sha(wrapper),
        actual_header_sha256=sha(runtime / 'complete_fc_math.h'),
        registration_header_sha256=sha(runtime / 'complete_fc.h'),
        compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
        compile_sha256=sha(a.output_dir / 'compile.json'), library_sha256=sha(library),
        export_report_sha256=hashlib.sha256(raw).hexdigest(),
        original_replay_report_sha256=sha(a.integer_replay_root / 'report.json'),
        original_ansi_replay_report_sha256=sha(a.ansi_replay_root / 'report.json'),
        synthetic_control_groups=controls, synthetic_control_codes=control_codes, cases=results,
        versions={n: importlib.metadata.version(n) for n in ['numpy', 'ai-edge-litert']},
        all_passed=all(g['exact'] for g in controls) and all(all(c[n] for n in
            ['fc_exact', 'downstream_control_exact', 'downstream_exact', 'batch_exact', 'repeat_exact']) for c in results),
        limits='Actual complete_fc_math.h C++ executed natively on Mac with original saved embedding and model weights. Exact comparisons to original actual LiteRT FC and downstream UINT8 trace, independent rational synthetic controls. Does not execute real TFLM Prepare, registration lifecycle, full model/P4 inference, optimized convolution, frontend, recognition/FAR, resources or candidate admission. Original models/traces unchanged, no PCM IO, optimizer, freshdev, acceptance or board writes.')
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['all_passed'], 'FC replay failed; preserved report/arrays in ' + str(a.output_dir)
    print(json.dumps(dict(all_passed=True, control_codes=control_codes, steps=sum(c['steps'] for c in results))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--integer-replay-root', type=Path, required=True)
    parser.add_argument('--ansi-replay-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    replay(parser.parse_args())
