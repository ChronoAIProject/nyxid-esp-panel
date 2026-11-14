"""Execute the real FC registration, installed TFLM Prepare and custom Eval.

Only the host allocation/tensor context and unused ESP timer are supplied by
the witness harness. This is not a MicroInterpreter or P4 runtime test.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SOURCES = [
    'tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc',
    'tensorflow/lite/micro/kernels/fully_connected_common.cc',
    'tensorflow/lite/micro/kernels/kernel_util.cc',
    'tensorflow/lite/micro/micro_context.cc',
    'tensorflow/lite/micro/micro_log.cc', 'tensorflow/lite/micro/debug_log.cc',
    'tensorflow/lite/micro/memory_helpers.cc',
    'tensorflow/lite/kernels/kernel_util.cc',
    'tensorflow/lite/kernels/internal/quantization_util.cc',
    'tensorflow/lite/kernels/internal/common.cc',
    'tensorflow/lite/core/c/common.cc',
    'tensorflow/lite/kernels/internal/portable_tensor_utils.cc',
    'tensorflow/lite/kernels/internal/reference/portable_tensor_utils.cc',
    'tensorflow/lite/kernels/internal/tensor_ctypes.cc',
]


class Config(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ['batches', 'depth', 'input_zero',
                'filter_zero', 'output_zero', 'activation', 'variant']] + [
                (n, ctypes.c_float) for n in ['input_scale', 'filter_scale', 'bias_scale', 'output_scale']]


class Receipt(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ['prepare_status', 'invoke_status',
                'temporary_count', 'activation_min', 'activation_max', 'input_zero', 'output_zero']] + [
                ('effective_scale', ctypes.c_double), ('persistent_bytes', ctypes.c_size_t)]


def replay(a):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    repo = Path(__file__).resolve().parents[3]
    a.output_dir.mkdir(exist_ok=False)
    component = repo / 'managed_components/espressif__esp-tflite-micro'
    runtime = repo / 'research/panel_wake_runtime'
    timer = a.output_dir / 'esp_timer.h'
    timer.write_text('#pragma once\n#include <stdint.h>\nextern "C" int64_t esp_timer_get_time();\n')
    wrapper = runtime / 'complete_fc_registration_host_v1.cc'
    library = a.output_dir / 'libcomplete_fc_registration.dylib'
    compiler = shutil.which('clang++')
    command = [compiler, '-std=c++17', '-O2', '-dynamiclib', '-fvisibility=hidden',
        '-Wl,-dead_strip', '-DTF_LITE_STATIC_MEMORY', '-DTF_LITE_DISABLE_X86_NEON',
        '-DESP_NN=0', '-DNDEBUG', '-I' + str(a.output_dir), '-I' + str(runtime)] + [
        '-I' + str(component / p) for p in ['.', 'third_party/gemmlowp',
                'third_party/flatbuffers/include', 'third_party/ruy']] + [str(wrapper)] + [
        str(component / p) for p in SOURCES] + ['-o', str(library)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (a.output_dir / 'compile.json').write_text(json.dumps(vars(compiled), indent=2) + '\n')
    compiled.check_returncode()
    dylib = ctypes.CDLL(str(library))
    fc = dylib.complete_fc_registration_host_v1
    p8, p32 = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)
    fc.argtypes = [p8, p8, p32, p8, ctypes.POINTER(Config), ctypes.POINTER(Receipt)]
    fc.restype = ctypes.c_int
    def execute(x, w, b, config):
        x, w = np.ascontiguousarray(x, dtype=np.int8), np.ascontiguousarray(w, dtype=np.int8)
        b = None if b is None else np.asarray([b], np.int32)
        output = np.full(config.batches, 101, np.int8)
        receipt = Receipt()
        assert fc(x.ctypes.data_as(p8), w.ctypes.data_as(p8),
                  None if b is None else b.ctypes.data_as(p32), output.ctypes.data_as(p8),
                  ctypes.byref(config), ctypes.byref(receipt)) == 0
        return output, {n: getattr(receipt, n) for n, _ in Receipt._fields_}

    ref = subprocess.check_output(['git', 'rev-parse', a.gpu_ref], text=True).strip()
    raw = subprocess.check_output(['git', 'show', ref + ':research/wakeword_p4/results/4090-hey-broad-equalized-export-result-v1.json'])
    export = json.loads(raw)
    prior = json.loads((a.integer_replay_root / 'report.json').read_text())
    cases = []
    for target in ['hey_willow', 'hey_nyx']:
        arm = next(v for v in export['arms'] if v['target'] == target)
        case = next(v for v in prior['cases'] if v['target'] == target)
        path = a.payload_root / arm['path']
        assert sha(path) == arm['sha256']
        assert sha(a.integer_replay_root / case['arrays']) == case['arrays_sha256']
        model = Interpreter(model_content=path.read_bytes(), experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        model.allocate_tensors()
        details = {d['index']: d for d in model.get_tensor_details()}
        op, = [o for o in arm['ops'] if o['op'] == 'FULLY_CONNECTED']
        embedding_id, weight_id, bias_id = op['inputs']
        output_id, = op['outputs']
        weights = model.get_tensor(weight_id).copy().reshape(-1)
        bias, = model.get_tensor(bias_id).reshape(-1)
        iscale, izero = details[embedding_id]['quantization']
        wscale, wzero = details[weight_id]['quantization']
        bscale, bzero = details[bias_id]['quantization']
        oscale, ozero = details[output_id]['quantization']
        assert bzero == wzero == 0 and len(weights) == 1800
        with np.load(a.integer_replay_root / case['arrays'], allow_pickle=False) as saved:
            inputs = saved[f'tensor_{embedding_id}'].copy().reshape(-1, 1800)
            expected = saved[f'tensor_{output_id}'].copy().reshape(-1)
        config = Config(len(inputs), 1800, izero, wzero, ozero, op['fused_activation'], 0,
                        iscale, wscale, bscale, oscale)
        first, r1 = execute(inputs, weights, bias, config)
        second, r2 = execute(inputs, weights, bias, config)
        arrays = a.output_dir / (target + '-registered-fc.npz')
        np.savez_compressed(arrays, expected=expected, actual=first, repeat=second)
        expected_scale = float(iscale) * float(wscale) / float(oscale)
        passed = np.array_equal(first, expected) and np.array_equal(first, second) and r1 == r2 and (
            r1['prepare_status'] == r1['invoke_status'] == r1['temporary_count'] == 0) and (
            r1['effective_scale'] == expected_scale and r1['input_zero'] == izero and r1['output_zero'] == ozero)
        cases.append(dict(target=target, steps=len(inputs), receipt=r1, repeat_receipt=r2,
            fc_unequal_steps=int(np.count_nonzero(first != expected)), passed=bool(passed),
            arrays=arrays.name, arrays_sha256=sha(arrays), model_sha256=sha(path)))

    controls = []
    x = np.asarray([[-3], [-1], [1], [3]], np.int8)
    w = np.asarray([1], np.int8)
    for name, variant, bias, depth in [
        ('optional-bias', 0, None, 1), ('present-bias', 0, 2, 1),
        ('multi-output-rejected', 1, 0, 1), ('non-scalar-bias-rejected', 2, 0, 1),
        ('wrong-output-type-rejected', 3, 0, 1), ('changed-invoke-shape-rejected', 4, 0, 1),
        ('per-channel-rejected', 5, 0, 1), ('overflow-risk-bias-rejected', 0, 2**31-1, 1)]:
        cfg = Config(4, depth, 0, 0, 0, 0, variant, .5, 1., .5, 1.)
        output, receipt = execute(x, w, bias, cfg)
        if name in ['optional-bias', 'present-bias']:
            expected = [-2, -1, 1, 2] if bias is None else [-1, 1, 2, 3]
            passed = receipt['prepare_status'] == receipt['invoke_status'] == 0 and output.tolist() == expected
        elif variant == 4:
            passed = receipt['prepare_status'] == 0 and receipt['invoke_status'] != 0 and np.all(output == 101)
        else:
            passed = receipt['prepare_status'] != 0 and receipt['invoke_status'] == -1 and np.all(output == 101)
        passed = passed and receipt['temporary_count'] == 0
        controls.append(dict(name=name, receipt=receipt, output=output.tolist(), passed=bool(passed)))
    report = dict(schema='mac-complete-fc-registration-replay-v1', gpu_ref=ref,
        script_sha256=sha(__file__), wrapper_sha256=sha(wrapper),
        production_headers={p: sha(runtime / p) for p in ['complete_fc.h', 'complete_fc_math.h']},
        installed_source_sha256={p: sha(component / p) for p in SOURCES},
        timer_stub_sha256=sha(timer), library_sha256=sha(library), compile_sha256=sha(a.output_dir / 'compile.json'),
        original_replay_report_sha256=sha(a.integer_replay_root / 'report.json'),
        export_report_sha256=hashlib.sha256(raw).hexdigest(), cases=cases, controls=controls,
        all_passed=all(c['passed'] for c in cases + controls),
        limits='Real complete_fc.h Init/Prepare/Invoke with actual installed TFLM registration/standard Prepare/activation calculation and eval tensor helpers, compiled ESP_NN=0 (Prepare source is unchanged by this flag). Host tensor/allocator context and unused zero timer supplied by harness. Actual source kernels are not shimmed. Model cases use original saved embeddings and exported scalar quantization metadata. No full MicroInterpreter arena, whole model/P4 inference, convolution, frontend, recognition/FAR/resource/candidate admission, PCM IO, optimizer, freshdev, acceptance or board writes.')
    (a.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    assert report['all_passed'], 'Preserved failed lifecycle report in ' + str(a.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--integer-replay-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    replay(parser.parse_args())
