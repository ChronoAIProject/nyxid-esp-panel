"""Lock completed final exports, then run registered full-wave numerical screens."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from collect_complete_development_v1 import runtime_identity_files

ROOT = Path('research/wakeword_p4')
PLAN = ROOT/'results/4090-hey-broad-trained-quantization-plan-v1.json'
LOCK = ROOT/'results/4090-hey-broad-trained-quantization-execution-v1.json'
OUT = Path('.local/wakeword/hey-broad-trained-quantization-v1')
PACKAGES = ('tensorflow', 'numpy', 'soundfile', 'ai-edge-litert', 'pymicro-features')


def published(path):
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:'+str(path)]) == path.read_bytes()


def completed(target, training):
    arm = training['arms'][target]
    folder = Path(arm['output'])
    receipt = json.loads((folder/'formal-execution-receipt.json').read_text())
    assert receipt['target'] == target and receipt['steps'] == 120000
    assert receipt['training_lock_sha256'] == sha(ROOT/'results/4090-hey-broad-formal-training-lock-v1.json')
    assert all(receipt[k] for k in ('finite_weights', 'finite_logged_loss', 'frozen_bn_exact'))
    assert receipt['model_sha256'] == sha(folder/'hey_nyx.tflite')
    assert receipt['training_report_sha256'] == sha(folder/'training.json')
    assert receipt['actual_initial_inventory_sha256'] == sha(folder/'actual-initial-inventory.json')
    report = json.loads((folder/'training.json').read_text())
    assert report['selected_step'] == 120000 and report['parameters'] == 24301
    assert report['batchnorm_initial'] == report['batchnorm_selected']
    trace = folder/'actual-batch-schedule.jsonl'
    assert report['batch_schedule']['actual_trace_sha256'] == sha(trace)
    expected = [json.loads(x) for x in Path(arm['schedule']).read_text().splitlines()]
    actual = [json.loads(x) for x in trace.read_text().splitlines()]
    assert len(actual) == len(expected) == 120100
    for i, (a, b) in enumerate(zip(actual, expected), 1):
        assert a['index'] == b['index'] == i and a['kind'] == b['kind'] == ('train' if i <= 120000 else 'ptq')
        assert a['matches_expected'] and a['tensor_sha256'] == b['tensor_sha256']
    return folder


def lock():
    published(PLAN)
    plan = json.loads(PLAN.read_text())
    training_path = ROOT/'results/4090-hey-broad-formal-training-lock-v1.json'
    assert sha(training_path) == plan['training_lock_sha256']
    training = json.loads(training_path.read_text())
    # Check completion before creating an execution lock or importing TensorFlow.
    folders = {t: completed(t, training) for t in plan['targets']}
    paths = [PLAN, training_path, Path(__file__),
             Path(__file__).with_name('plan_hey_broad_trained_quantization_v1.py'),
             Path(__file__).with_name('collect_complete_development_v1.py'),
             Path(__file__).with_name('complete_feature_stream_v1.py'),
             Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),
             ROOT/'training/features.py', Path(plan['ledger_path'])]
    paths += runtime_identity_files()
    paths += list(Path(training['upstream']).rglob('*.py'))
    # TensorFlow native kernels and Python wrappers are execution dependencies too.
    import importlib.util
    tf_root = Path(importlib.util.find_spec('tensorflow').origin).parent
    paths += [p for p in tf_root.rglob('*') if p.is_file() and p.suffix in ('.py', '.so')]
    paths += [Path(p) for p in training['files'] if Path(p).suffix == '.py']
    for folder in folders.values():
        paths += [folder/n for n in ('formal-execution-receipt.json', 'training.json',
                  'best.weights.h5', 'hey_nyx.tflite', 'actual-initial-inventory.json', 'actual-batch-schedule.jsonl')]
    paths += [Path(r['path']) for r in plan['requests']]
    files = {str(p): sha(p) for p in sorted(set(paths))}
    for row in plan['requests']:
        assert files[row['path']] == row['sha256']
    assert files[plan['ledger_path']] == plan['ledger_sha256']
    result = dict(schema='hey-broad-trained-quantization-execution-v1', files=files,
                  models={t: str(p) for t, p in folders.items()},
                  versions={p: importlib.metadata.version(p) for p in PACKAGES})
    with LOCK.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


def errors(value):
    return dict(mean=float(np.mean(value)), p99=float(np.percentile(value, 99, method='linear')),
                maximum=float(np.max(value)))


def execute():
    published(LOCK)
    execution = json.loads(LOCK.read_text())
    for path, expected in execution['files'].items():
        assert sha(path) == expected, path
    for name, version in execution['versions'].items():
        assert importlib.metadata.version(name) == version
    plan = json.loads(PLAN.read_text())
    training = json.loads((ROOT/'results/4090-hey-broad-formal-training-lock-v1.json').read_text())
    for target in plan['targets']:
        completed(target, training)
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    tf = helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    sys.path.insert(0, str((ROOT/'training').resolve()))
    from features import features
    cases = []
    for target, folder_name in execution['models'].items():
        folder = Path(folder_name)
        model, offline = helper.create(225), helper.create(225)
        for instance in (model, offline):
            instance.load_weights(folder/'best.weights.h5')
            for layer in instance.layers:
                cell = getattr(layer, 'cell', layer)
                if isinstance(cell, tf.keras.layers.BatchNormalization):
                    layer.trainable = cell.trainable = False
        original = [v.numpy().copy() for v in offline.variables]
        assert all(np.array_equal(a, b.numpy()) for a, b in zip(original, model.variables, strict=True))
        stream = utils.to_streaming_inference(model, {'stride': 3, 'spectrogram_length': 225}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1, 3, 40), np.float32), training=False)
        states = [v for v in stream.variables if 'states' in v.name]
        assert len(states) == 6
        infer = tf.function(lambda x: stream(x, training=False), input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
        window = tf.function(lambda x: offline(x, training=False), input_signature=[tf.TensorSpec([1, 225, 40], tf.float32)])
        for index, row in enumerate(plan['requests']):
            pcm, rate = sf.read(row['path'], dtype='int16')
            assert rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
            assert __import__('hashlib').sha256(pcm.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
            raw = features(pcm)
            padded = np.pad(raw, ((0, (-len(raw)) % 3), (0, 0)))
            values = np.clip(padded.astype(np.float32)/25.6, 0, 26)
            floats, integers = [], []
            for repeat in range(2):
                for state in states:
                    state.assign(tf.zeros_like(state))
                floats.append(np.asarray([float(infer(values[None, j:j+3]).numpy()[0, 0]) for j in range(0, len(values), 3)], np.float32))
                interpreter = Interpreter(model_path=str(folder/'hey_nyx.tflite'), experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
                interpreter.allocate_tensors()
                inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
                assert list(inp['shape']) == [1, 3, 40] and inp['dtype'] == np.int8
                assert list(out['shape']) == [1, 1] and out['dtype'] == np.uint8
                scale, zero = inp['quantization']; oscale, ozero = out['quantization']
                assert scale > 0 and oscale > 0
                q = np.clip(np.rint(values/scale)+zero, -128, 127).astype(np.int8)
                result = []
                for j in range(0, len(q), 3):
                    interpreter.set_tensor(inp['index'], q[None, j:j+3]); interpreter.invoke()
                    result.append(int(interpreter.get_tensor(out['index'])[0, 0]))
                integers.append(np.asarray(result, np.uint8))
            repeat_exact = np.array_equal(*floats) and np.array_equal(*integers)
            ends = list(range(225, len(values)+1, 3))
            nominal = np.asarray([float(window(values[None, end-225:end]).numpy()[0, 0]) for end in ends])
            phase = np.asarray([float(window(np.concatenate((values[end-224:end], np.zeros((1, 40), np.float32)))[None]).numpy()[0, 0]) for end in ends])
            actual = floats[0].astype(np.float64)
            dequantized = (integers[0].astype(np.float64)-ozero)*oscale
            assert len(actual) == len(dequantized) == len(values)//3
            signed = dequantized-actual
            smooth = np.convolve(signed, np.ones(5)/5, mode='valid')
            e = errors(np.abs(smooth)); phase_error = float(np.max(np.abs(phase-actual[74:])))
            gates = plan['numerical_gates']
            passed = bool(repeat_exact and np.isfinite(actual).all() and
                          phase_error <= gates['float_offline_stream_max_abs'] and
                          e['mean'] <= gates['per_input_five_step_mean_abs_error_max'] and
                          e['p99'] <= gates['per_input_five_step_p99_abs_error_max'] and
                          e['maximum'] <= gates['per_input_five_step_max_abs_error_max'])
            artifact = OUT/f'{target}-{index:03}.npz'
            clocks = (3*np.arange(1, len(actual)+1)+2)*160
            np.savez(artifact, raw=raw, quantized_input=q, float=actual, uint8=integers[0],
                     float_repeat=floats[1], uint8_repeat=integers[1], offline_nominal=nominal,
                     offline_phase=phase, support_unclamped=clocks, sample_end=np.minimum(clocks, len(pcm)))
            case = dict(target=target, request_order=row['order'], repeat_exact=bool(repeat_exact),
                        phase_max_error=phase_error, smoothed_abs_error=e, raw_abs_error=errors(np.abs(signed)),
                        signed_mean=float(np.mean(signed)), passed=passed, arrays_path=str(artifact),
                        arrays_sha256=sha(artifact), input_quantization=[scale, zero], output_quantization=[oscale, ozero])
            cases.append(case)
            with (OUT/'cases.jsonl').open('a') as output:
                output.write(json.dumps(case)+'\n')
            print(json.dumps(dict(target=target, completed=index+1, passed=passed)), flush=True)
        assert all(np.array_equal(a, b.numpy()) for a, b in zip(original, offline.variables, strict=True))
    result = dict(schema='hey-broad-trained-quantization-result-v1', execution_sha256=sha(LOCK),
                  cases=cases, all_passed=all(c['passed'] for c in cases), limits=plan['limits'])
    with (ROOT/'results/4090-hey-broad-trained-quantization-result-v1.json').open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['lock', 'run'])
    args = parser.parse_args()
    lock() if args.action == 'lock' else execute()
