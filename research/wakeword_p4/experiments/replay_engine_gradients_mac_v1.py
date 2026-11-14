"""Replay the original zero-update engine probe from authenticated snapshots."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execution-mode', choices=('graph', 'eager'), default='graph')
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(20261114)
    tf.config.experimental.enable_op_determinism()
    root, out = args.payload_root.resolve(), args.output_dir.resolve()
    out.mkdir(exist_ok=False)
    assert sha(root / 'manifest.json') == '27d0a086a61cc6934e552eb5e97acb62a0e78f4c8a2be063a896f389e790b880'
    manifest = json.loads((root / 'manifest.json').read_text())
    mapping = {}
    for name, receipt in manifest['files'].items():
        path = root / name
        assert path.is_relative_to(root) and path.stat().st_size == receipt['bytes']
        assert sha(path) == receipt['sha256']
        assert receipt['path'] not in mapping
        mapping[receipt['path']] = path

    def mapped(path):
        return mapping[path]

    lock = json.loads((root / 'metadata/4090-engine-adversarial-gradient-execution-v1.json').read_text())
    result = json.loads((root / 'metadata/4090-engine-adversarial-gradient-result-v1.json').read_text())
    assert result['all_passed'] and result['optimizer_created'] is False
    assert result['execution_sha256'] == sha(root / 'metadata/4090-engine-adversarial-gradient-execution-v1.json')
    verified = {}
    for path, expected in lock['files'].items():
        needle = '.local/wakeword/microWakeWord/'
        if needle in path and path.endswith('.py'):
            relative = path.split(needle, 1)[1]
            assert sha(args.upstream / relative) == expected, relative
            verified[relative] = expected
    assert len(verified) == 19
    versions = {name: importlib.metadata.version(name) for name in lock['versions']}
    assert versions == lock['versions']
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    namespace = dict(argparse=argparse, mixednet=mixednet)
    tree = ast.parse((root / 'source/probe_hey_context_transfer_shapes_v1.py').read_text())
    nodes = [node for node in tree.body if
        isinstance(node, ast.FunctionDef) and node.name == 'create' or
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ARCH' for t in node.targets)]
    assert len(nodes) == 2
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<original-architecture>', 'exec'), namespace)
    probe = ast.parse((root / 'source/probe_engine_adversarial_gradients_v1.py').read_text())
    reverse = [node for node in ast.walk(probe) if isinstance(node, ast.FunctionDef) and node.name == 'reverse']
    assert len(reverse) == 1
    reversal = dict(tf=tf)
    exec(compile(ast.fix_missing_locations(ast.Module(body=reverse, type_ignores=[])),
                 '<original-reversal>', 'exec'), reversal)
    aux = json.loads((root / 'metadata/4090-engine-adversarial-input-result-v1.json').read_text())['batches']
    assert [b['step'] for b in aux] == list(range(1, 51))
    cases, snapshots = [], []

    def comparison(actual, expected):
        a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        delta = np.abs(a - b)
        return dict(max_abs=float(delta.max(initial=0)),
            max_tolerance_fraction=float((delta / (1e-4 + 1e-4 * np.abs(b))).max(initial=0)),
            passed=bool(np.all(delta <= 1e-4 + 1e-4 * np.abs(b))))

    for target in result['targets']:
        name = target['target']
        assert name in ('hey_nyx', 'hey_willow') and len(target['cases']) == 50
        init_path = mapped(target['initialization_path'])
        assert sha(init_path) == target['initialization_sha256']
        init = json.loads(init_path.read_text())
        model = namespace['create'](225)
        model.load_weights(mapped(init['initial_h5_path']))
        assert sha(mapped(init['initial_h5_path'])) == init['initial_h5_sha256']
        for layer in model.layers:
            cell = getattr(layer, 'cell', layer)
            if isinstance(cell, tf.keras.layers.BatchNormalization):
                layer.trainable = cell.trainable = False
        model_vars = list(model.variables)
        with np.load(mapped(init['state_path']), allow_pickle=False) as original, np.load(
                mapped(target['end_state_path']), allow_pickle=False) as ending:
            assert len(model_vars) + 2 == len(init['inventory']) == 33
            domain_values = [original[f'v{i}'].copy() for i in (31, 32)]
            for i, receipt in enumerate(init['inventory']):
                data = original[f'v{i}']
                assert list(data.shape) == receipt['shape'] and str(data.dtype) == receipt['dtype']
                assert hashlib.sha256(data.tobytes()).hexdigest() == receipt['sha256']
                assert np.array_equal(data, ending[f'v{i}'])
                if i < len(model_vars):
                    assert model_vars[i].numpy().tobytes() == data.tobytes(), (name, i)
        kernel = tf.Variable(domain_values[0], dtype=tf.float32)
        bias = tf.Variable(domain_values[1], dtype=tf.float32)
        variables = list(model.trainable_variables) + [kernel, bias]
        assert len(variables) == 17
        for var, receipt in zip(variables, init['trainable_inventory'], strict=True):
            data = var.numpy()
            assert list(data.shape) == receipt['shape'] and str(data.dtype) == receipt['dtype']
            assert hashlib.sha256(data.tobytes()).hexdigest() == receipt['sha256']
        head = model.layers[-1]
        wake, encoder = init['wake_indices'], init['encoder_indices']
        assert wake == [13, 14] and encoder == list(range(13))
        embeddings = tf.keras.Model(model.inputs, head.input)
        all_vars = model_vars + [kernel, bias]
        initial = [v.numpy().tobytes() for v in all_vars]
        snapshots.append(dict(target=name, all33_same_instance_arrays_byte_exact=True,
            original_h5_sha256=init['initial_h5_sha256']))

        def calculate(x, y, weights, ax, labels):
            with tf.GradientTape(persistent=True) as tape:
                logits = tf.linalg.matmul(embeddings(x, training=False), head.kernel) + head.bias
                S = tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=logits), [-1]) * weights)
                features = embeddings(ax, training=False)
                dz = tf.linalg.matmul(features, kernel) + bias
                D = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=dz))
                r0 = tf.linalg.matmul(reversal['reverse'](features, 0.), kernel) + bias
                r1 = tf.linalg.matmul(reversal['reverse'](features, 1.), kernel) + bias
                R0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=r0))
                R1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=r1))
            payload = dict(main_logits=logits, domain_logits=dz, domain_features=features,
                           losses=tf.stack([S, D, R0, R1]))
            for tag, loss, missing in zip(('S', 'D', 'R0', 'R1'), (S, D, R0, R1),
                                         ([15, 16], wake, wake, wake), strict=True):
                gradients = tape.gradient(loss, variables)
                assert [i for i, g in enumerate(gradients) if g is None] == missing
                payload.update({f'{tag}_{i}': g for i, g in enumerate(gradients) if g is not None})
            del tape
            return payload

        if args.execution_mode == 'graph':
            calculate = tf.function(calculate)

        for index, case in enumerate(target['cases']):
            assert case['step'] == index + 1 and case['aux_arrays_sha256'] == aux[index]['arrays_sha256']
            path, aux_path = mapped(case['arrays_path']), mapped(aux[index]['arrays_path'])
            assert sha(path) == case['arrays_sha256'] and sha(aux_path) == case['aux_arrays_sha256']
            with np.load(path, allow_pickle=False) as saved, np.load(aux_path, allow_pickle=False) as auxiliary:
                args_tf = tuple(tf.constant(value) for value in
                    (saved['main_x'], saved['main_y'], saved['main_weights'], auxiliary['x'], auxiliary['engine']))
                repeats = [{key: value.numpy() for key, value in calculate(*args_tf).items()} for _ in range(2)]
                actual = repeats[0]
                assert all(actual[k].tobytes() == repeats[1][k].tobytes() for k in actual)
                assert all(np.isfinite(v).all() for v in actual.values())
                assert case['none_indices'] == [[15, 16], wake, wake, wake]
                for i in encoder:
                    assert not np.count_nonzero(actual[f'R0_{i}'])
                    assert np.array_equal(actual[f'R1_{i}'], -actual[f'D_{i}'])
                for i in (15, 16):
                    assert np.array_equal(actual[f'R0_{i}'], actual[f'D_{i}'])
                    assert np.array_equal(actual[f'R1_{i}'], actual[f'D_{i}'])
                errors = {key: comparison(actual[key], saved[key]) for key in actual if key != 'domain_features'}
                logits = saved['domain_logits'].astype(np.float64)
                p = np.exp(logits - logits.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                p[np.arange(33), auxiliary['engine']] -= 1
                bias_error = float(np.max(np.abs(p.mean(axis=0) - saved['D_16'])))
                assert bias_error <= 1e-6
                file = f'{name}-{index + 1:03}.npz'
                np.savez_compressed(out / file, **actual)
                row = dict(target=name, step=index + 1, cpu_repeats_byte_exact=True,
                    reverse_sign_zero_and_none_patterns_exact=True,
                    analytic_saved_domain_bias_max_abs=bias_error,
                    errors=errors, all_numeric_comparisons_passed=all(v['passed'] for v in errors.values()),
                    arrays=file, arrays_sha256=sha(out / file))
                cases.append(row)
                if (index + 1) % 10 == 0:
                    print(json.dumps(dict(target=name, batches=index + 1,
                        numeric_failures_so_far=sum(not c['all_numeric_comparisons_passed'] for c in cases))), flush=True)
            assert [v.numpy().tobytes() for v in all_vars] == initial
        tf.keras.backend.clear_session()
    assert len(cases) == 100
    report = dict(schema='mac-engine-gradient-original-snapshot-replay-v1',
        execution_mode=args.execution_mode,
        reviewer_sha256=sha(__file__), manifest_sha256=sha(root / 'manifest.json'),
        producer_commit=manifest['producer_source_commit'], result_commit=manifest['result_commit'],
        runtime_versions=versions, original_upstream_python_sha256=verified,
        snapshot_checks=snapshots, cases=cases, actual_batches=100, cpu_repeats=2,
        all_initial_and_end_variables_byte_exact=True,
        all_numeric_comparisons_passed=all(c['all_numeric_comparisons_passed'] for c in cases),
        numerical_rule='Pointwise abs(delta)<=1e-4+1e-4*abs(saved); saved analytic domain bias abs<=1e-6.',
        optimizer_created=False, updates=0, audio_accessed=False, candidate_qualified=False,
        limits='Original initial snapshots and original training-case tensors only. '
               'CPU retraces the original architecture/reversal with unchanged variables; '
               'no new training, coefficients, thresholds, recognition, fresh data or P4 acceptance.')
    write(out / 'report.json', report)
    print(json.dumps({k: report[k] for k in ('actual_batches', 'cpu_repeats', 'all_numeric_comparisons_passed')}))
    return 0 if report['all_numeric_comparisons_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
