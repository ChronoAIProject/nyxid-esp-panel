"""Prospective native replay of four fixed-grid models on two old witnesses.

This does not select candidates or replace the complete540 numerical check.
There is no audio IO, frontend regeneration, optimizer, export or board write.
"""
import argparse
import ast
import ctypes
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


ROOT = 'research/wakeword_p4/'
PLAN = Path(__file__).resolve().parents[1] / 'results/mac-bounded-consistency-fixed-grid-witness-replay-plan-v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(ref, name):
    return subprocess.check_output(['git', 'show', ref + ':' + ROOT + name])


def make_plan(a):
    import numpy as np
    ref = subprocess.check_output(['git', 'rev-parse', a.gpu_ref], text=True).strip()
    names = ['results/4090-hey-broad-bounded-consistency-fixed-grid-compile-result-v1.json',
             'results/4090-hey-broad-bounded-consistency-fixed-grid-compile-plan-v1.json',
             'results/4090-hey-broad-bounded-consistency-formal-result-v1.json',
             'results/4090-hey-broad-equalized-screen-execution-v1.json',
             'results/4090-hey-broad-equalized-screen-result-v1.json',
             'experiments/probe_hey_context_transfer_shapes_v1.py',
             'experiments/probe_hey_broad_bounded_float_v1.py']
    raw = {n: source(ref, n) for n in names}
    metadata, formal, equalized = [json.loads(raw['results/' + n]) for n in [
        '4090-hey-broad-bounded-consistency-fixed-grid-compile-result-v1.json',
        '4090-hey-broad-bounded-consistency-formal-result-v1.json',
        '4090-hey-broad-equalized-screen-result-v1.json']]
    inputs = []
    for target, order in [('hey_willow', 14), ('hey_nyx', 48394)]:
        case, = [c for c in equalized['cases'] if (c['target'], c['request_order']) == (target, order)]
        path = a.old_feature_payload / case['arrays_path']
        assert sha(path) == case['arrays_sha256']
        with np.load(path, allow_pickle=False) as saved:
            features = saved['raw'].copy()
        inputs.append(dict(target=target, request_order=order, path=case['arrays_path'],
            arrays_sha256=case['arrays_sha256'], raw_dtype=str(features.dtype),
            raw_shape=list(features.shape), raw_sha256=hashlib.sha256(features.tobytes()).hexdigest()))
    arms = []
    for arm in metadata['arms']:
        run, = [c for c in formal['cases'] if (c['target'], c['condition']) == (arm['target'], arm['condition'])]
        final = run['final_repeat']
        assert run['steps'] == 10000 and final['weights_sha256'] == arm['final_h5_sha256']
        arms.append(dict(target=arm['target'], condition=arm['condition'], model_path=arm['path'],
            model_sha256=arm['sha256'], model_bytes=arm['bytes'],
            weights_path=final['weights_path'], weights_sha256=final['weights_sha256']))
    expected = {(target, condition) for target in ['hey_willow', 'hey_nyx']
                for condition in ['raw_logit_control', 'bounded_logit_consistency']}
    assert len(arms) == 4 and {(c['target'], c['condition']) for c in arms} == expected
    assert formal['all_passed'], 'All four original formal arms must be complete'
    registration = json.loads((a.registered_fc_root / 'report.json').read_text())
    assert registration['all_passed'] is True
    record = dict(schema='mac-bounded-consistency-fixed-grid-witness-replay-plan-v1', gpu_ref=ref,
        source_sha256={ROOT + n: hashlib.sha256(data).hexdigest() for n, data in raw.items()},
        script_sha256=sha(__file__), inputs=inputs, arms=arms,
        registered_fc_report_sha256=sha(a.registered_fc_root / 'report.json'),
        registered_fc_library_sha256=registration['library_sha256'],
        policy='Exactly four matched bounded-family fixed-final10000/fixed-grid models (raw_logit_control and bounded_logit_consistency), each on its target old predetermined witness (Willow14 or Nyx48394). Both conditions use identical target input; no scoring-based case choice. Two fresh actual LiteRT whole-input runs and two explicit-reset float streams; fixed phase224+zero. Compare actual native registered FC on saved actual embedding to original actual FC. Preserve all outputs, failures and unchanged .02/.1/.25 numerical bounds. At most comparison to published same-case metadata, not all540 array verification or candidate admission. No freshdev, independent acceptance, waveform IO, resampling, optimizer, export or board operations.')
    with PLAN.open('x') as f:
        f.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(plan=str(PLAN), sha256=sha(PLAN), arms=len(arms), inputs=len(inputs))))


def replay(a):
    import numpy as np
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    from ai_edge_litert import schema_py_generated as schema
    from replay_complete_fc_registration_mac_v1 import Config, Receipt

    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    lock = json.loads(PLAN.read_text())
    repo = Path(__file__).resolve().parents[3]
    relative = str(PLAN.relative_to(repo))
    assert subprocess.check_output(['git', 'show', 'origin/research/hey-nyx-4090:' + relative]) == PLAN.read_bytes()
    assert sha(__file__) == lock['script_sha256']
    ref = lock['gpu_ref']
    raw = {}
    for path, expected in lock['source_sha256'].items():
        data = subprocess.check_output(['git', 'show', ref + ':' + path])
        assert hashlib.sha256(data).hexdigest() == expected
        raw[path] = data
    origin = json.loads(raw[ROOT + 'results/4090-hey-broad-equalized-screen-execution-v1.json'])
    verified = {}
    for name, expected in origin['files'].items():
        needle = '.local/wakeword/microWakeWord/'
        if needle in name and name.endswith('.py'):
            relative = name.split(needle, 1)[1]
            assert sha(a.upstream / relative) == expected
            verified[relative] = expected
    assert len(verified) == 20
    for name, expected in origin['versions'].items():
        assert importlib.metadata.version(name) == expected, name
    sys.path.insert(0, str(a.upstream))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    namespace = dict(argparse=argparse, mixednet=mixednet, np=np)
    architecture = ast.parse(raw[ROOT + 'experiments/probe_hey_context_transfer_shapes_v1.py'])
    bounded = ast.parse(raw[ROOT + 'experiments/probe_hey_broad_bounded_float_v1.py'])
    nodes = [n for n in architecture.body if
        (isinstance(n, ast.FunctionDef) and n.name == 'create') or
        (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ARCH' for t in n.targets))]
    nodes += [n for n in bounded.body if isinstance(n, ast.FunctionDef) and n.name == 'bounded']
    assert len(nodes) == 3
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<bound-fixed-grid-float-helpers>', 'exec'), namespace)
    registration_path = a.registered_fc_root / 'report.json'
    assert sha(registration_path) == lock['registered_fc_report_sha256']
    registration = json.loads(registration_path.read_text())
    for name, expected in registration['production_headers'].items():
        assert sha(repo / 'research/panel_wake_runtime' / name) == expected
    library = a.registered_fc_root / 'libcomplete_fc_registration.dylib'
    assert sha(library) == lock['registered_fc_library_sha256']
    dylib = ctypes.CDLL(str(library))
    fc = dylib.complete_fc_registration_host_v1
    p8, p32 = ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_int32)
    fc.argtypes = [p8, p8, p32, p8, ctypes.POINTER(Config), ctypes.POINTER(Receipt)]
    fc.restype = ctypes.c_int
    recovered = json.loads(raw[ROOT + 'results/4090-hey-broad-bounded-consistency-fixed-grid-compile-result-v1.json'])['arms']
    templates = json.loads(raw[ROOT + 'results/4090-hey-broad-bounded-consistency-fixed-grid-compile-plan-v1.json'])['arms']
    exports = []
    for recovered_arm in recovered:
        template, = [entry['template'] for entry in templates if
                     (entry['target'], entry['condition']) ==
                     (recovered_arm['target'], recovered_arm['condition'])]
        assert recovered_arm['template_sha256'] == template['sha256']
        exports.append(dict(recovered_arm, ops=template['ops']))
    a.output_dir.mkdir(exist_ok=False)
    cases = []
    for arm in lock['arms']:
        target, condition = arm['target'], arm['condition']
        case, = [c for c in lock['inputs'] if c['target'] == target]
        metadata, = [e for e in exports if (e['target'], e['condition']) == (target, condition)]
        weights_path, model_path = [a.payload_root / arm[n] for n in ['weights_path', 'model_path']]
        assert sha(weights_path) == arm['weights_sha256']
        assert sha(model_path) == arm['model_sha256'] and model_path.stat().st_size == arm['model_bytes']
        feature_path = a.old_feature_payload / case['path']
        assert sha(feature_path) == case['arrays_sha256']
        with np.load(feature_path, allow_pickle=False) as saved:
            features = saved['raw'].copy()
        assert (str(features.dtype), list(features.shape), hashlib.sha256(features.tobytes()).hexdigest()) == (
            case['raw_dtype'], case['raw_shape'], case['raw_sha256'])
        values = np.clip(np.pad(features, ((0, (-len(features)) % 3), (0, 0))).astype(np.float32) / 25.6, 0, 26)
        models, originals, snapshots = [], [], []
        for _ in range(2):
            original = namespace['create'](225)
            original.load_weights(weights_path)
            for layer in original.layers:
                cell = getattr(layer, 'cell', layer)
                if isinstance(cell, tf.keras.layers.BatchNormalization):
                    layer.trainable = cell.trainable = False
            snapshots.append([v.numpy().copy() for v in original.variables])
            originals.append(original)
            models.append(namespace['bounded'](original, tf))
        stream = utils.to_streaming_inference(models[0], {'stride': 3, 'spectrogram_length': 225}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1, 3, 40), np.float32), training=False)
        states = [v for v in stream.variables if 'states' in v.name]
        assert len(states) == 6
        infer = tf.function(lambda x: stream(x, training=False), input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
        offline = tf.function(lambda x: models[1](x, training=False), input_signature=[tf.TensorSpec([1, 225, 40], tf.float32)])
        floats = []
        for _ in range(2):
            for state in states:
                state.assign(tf.zeros_like(state))
            floats.append(np.asarray([float(infer(values[None, n:n+3]).numpy()[0, 0]) for n in range(0, len(values), 3)]))
        phase = np.asarray([float(offline(np.concatenate((values[end-224:end], np.zeros((1, 40), np.float32)))[None]).numpy()[0, 0]) for end in range(225, len(values)+1, 3)])
        phase_max = float(np.max(np.abs(phase - floats[0][74:])))
        fc_op, = [o for o in metadata['ops'] if o['op'] == 'FULLY_CONNECTED']
        embedding_id, weight_id, bias_id = fc_op['inputs']
        fc_id, = fc_op['outputs']
        integers, embeddings, fc_codes = [], [], []
        for repeat in range(2):
            interpreter = Interpreter(model_content=model_path.read_bytes(), experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                                      experimental_preserve_all_tensors=True, num_threads=1)
            interpreter.allocate_tensors()
            inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            assert tuple(inp['shape']) == (1, 3, 40) and inp['dtype'] == np.int8
            assert tuple(out['shape']) == (1, 1) and out['dtype'] == np.uint8
            scale, zero = inp['quantization']
            oscale, ozero = out['quantization']
            q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
            scores, embed, codes = [], [], []
            for n in range(0, len(q), 3):
                interpreter.set_tensor(inp['index'], q[None, n:n+3])
                interpreter.invoke()
                scores.append(int(interpreter.get_tensor(out['index'])[0, 0]))
                embed.append(interpreter.get_tensor(embedding_id).copy())
                codes.append(int(interpreter.get_tensor(fc_id)[0, 0]))
            integers.append(np.asarray(scores, np.uint8))
            embeddings.append(np.ascontiguousarray(np.concatenate(embed), dtype=np.int8))
            fc_codes.append(np.asarray(codes, np.int8))
        details = {d['index']: d for d in interpreter.get_tensor_details()}
        weights = np.ascontiguousarray(interpreter.get_tensor(weight_id).reshape(-1), dtype=np.int8)
        bias = np.ascontiguousarray(interpreter.get_tensor(bias_id).reshape(-1), dtype=np.int32)
        iscale, izero = details[embedding_id]['quantization']
        wscale, wzero = details[weight_id]['quantization']
        bscale, bzero = details[bias_id]['quantization']
        fscale, fzero = details[fc_id]['quantization']
        assert wzero == bzero == 0 and len(weights) == 1800 and len(bias) == 1
        config = Config(len(embeddings[0]), 1800, izero, wzero, fzero, fc_op['fused_activation'], 0,
                        iscale, wscale, bscale, fscale)
        receipt = Receipt()
        registered = np.empty(len(embeddings[0]), np.int8)
        assert fc(embeddings[0].ctypes.data_as(p8), weights.ctypes.data_as(p8), bias.ctypes.data_as(p32), registered.ctypes.data_as(p8), ctypes.byref(config), ctypes.byref(receipt)) == 0
        repeat_exact = all(np.array_equal(v[0], v[1]) for v in [floats, integers, embeddings, fc_codes])
        registered_exact = np.array_equal(registered, fc_codes[0]) and receipt.prepare_status == receipt.invoke_status == receipt.temporary_count == 0
        unchanged = all(np.array_equal(before, variable.numpy()) for model, snapshot in zip(originals, snapshots, strict=True) for before, variable in zip(snapshot, model.variables, strict=True))
        signed = (integers[0].astype(np.float64) - ozero) * oscale - floats[0]
        smooth = np.abs(np.convolve(signed, np.ones(5) / 5, mode='valid'))
        metrics = dict(mean=float(np.mean(smooth)), p99=float(np.quantile(smooth, .99)), maximum=float(np.max(smooth)))
        numerical = phase_max <= 1e-4 and metrics['mean'] <= .02 and metrics['p99'] <= .1 and metrics['maximum'] <= .25
        filename = target + '-' + condition + '.npz'
        np.savez_compressed(a.output_dir / filename, raw=features, quantized_input=q,
            float=floats[0], float_repeat=floats[1], uint8=integers[0], uint8_repeat=integers[1],
            offline_phase=phase, embedding=embeddings[0], embedding_repeat=embeddings[1],
            actual_fc=fc_codes[0], actual_fc_repeat=fc_codes[1], registered_fc=registered)
        result = dict(target=target, condition=condition, request_order=case['request_order'], steps=len(signed),
            model_sha256=arm['model_sha256'], final_h5_sha256=arm['weights_sha256'],
            actual_repeat_exact=bool(repeat_exact), float_offline_phase_max_abs=phase_max,
            original_variables_unchanged=bool(unchanged), smoothed_abs_error=metrics,
            witness_meets_numerical_bounds=bool(numerical), registered_fc_exact=bool(registered_exact),
            registered_fc_unequal_steps=int(np.count_nonzero(registered != fc_codes[0])),
            actual_registration_receipt={n: getattr(receipt, n) for n, _ in Receipt._fields_},
            operations_passed=bool(repeat_exact and unchanged and registered_exact and phase_max <= 1e-4),
            arrays=filename, arrays_sha256=sha(a.output_dir / filename))
        cases.append(result)
        print(json.dumps(result), flush=True)
    report = dict(schema='mac-bounded-consistency-fixed-grid-witness-replay-v1', plan_sha256=sha(PLAN), gpu_ref=ref,
        script_sha256=sha(__file__), upstream_python_sha256=verified,
        versions={n: importlib.metadata.version(n) for n in ['tensorflow', 'numpy', 'keras', 'h5py', 'ai-edge-litert']},
        cases=cases, all_operations_passed=all(c['operations_passed'] for c in cases),
        limits=lock['policy'] + ' Native Mac CPU2threads/actual finalH5/bounded float and actual LiteRT fixedgrid interpreter; real installed TFLM FC registration with host tensor/allocator context only. Whole model inference here uses LiteRT, not TFLM/P4. Four predetermined old witnesses only; no all540 qualification or recognition/resource/first-utterance acceptance claim.')
    with (a.output_dir / 'report.json').open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    assert report['all_operations_passed'], 'Preserved failed native witness execution in ' + str(a.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--gpu-ref')
    parser.add_argument('--payload-root', type=Path)
    parser.add_argument('--old-feature-payload', type=Path, required=True)
    parser.add_argument('--registered-fc-root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    make_plan(args) if args.action == 'plan' else replay(args)
