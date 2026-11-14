"""Read-only CPU quantization-contract diagnostic of all 80 final constants.

This isolates constant expressions, not the complete original training graph.
No learned activation inference, model export, optimizer, or audio operations.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parents[1]
REPO = BASE.parents[1]
PREFIX = 'research/wakeword_p4/'
PLAN = BASE / 'results/mac-all-final-constant-capture-plan-v1.json'
PARENT = BASE / 'results/mac-consistency-fixed-grid-witness-replay-plan-v1.json'
PRIMITIVE = 'experiments/hey_broad_fake_quant_primitives_v1.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tracked(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path])


def make_plan(args):
    parent = json.loads(PARENT.read_text())
    ref = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    primitive = tracked(ref, PREFIX + PRIMITIVE)
    record = dict(
        schema='mac-all-final-constant-capture-plan-v1',
        script_sha256=sha(Path(__file__).read_bytes()),
        parent_plan_sha256=sha(PARENT.read_bytes()),
        primitive_gpu_ref=ref, primitive_sha256=sha(primitive),
        arms=parent['arms'], source_gpu_ref=parent['gpu_ref'],
        source_sha256=parent['source_sha256'],
        settings=['default', 'no_constant_folding_or_arithmetic'],
        methods=['eager_captured_scale', 'graph_captured_scale', 'graph_runtime_scale'],
        constants_per_model=20,
        policy='All four original final10000 H5/compiled models locked by050bf88, all20 weight/bias constants each, no case selection. Independent CPU-only processes for default and both constant_folding/arithmetic disabled. Each uses unchanged original affine_ste with captured scales eagerly/in tf.function and runtime tensor scale arguments. Two repeats, actual H5 variables unchanged. Direct original weight, frozenBN folding and boundedhead parameter expressions, correct per-channel layout, allocated builtinref constant reads only. Save every value/dequant/quotient/reconstructed code/actual compiled code and all differences. Full sweep continues through differences; those are diagnostic observations, not revised numerical gates. No actual activation inference, gradients, optimizer, export, fresh data, development/acceptance, or board/audio IO. These isolated expression graphs need not have the same optimization as the original full training graph; causal propagation/recognition effects remain unproven.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(plan=str(PLAN), sha256=sha(PLAN.read_bytes()))))


def worker(args):
    import numpy as np
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType

    plan = json.loads(PLAN.read_text())
    assert tracked('origin/research/hey-nyx-4090', str(PLAN.relative_to(REPO))) == PLAN.read_bytes()
    assert sha(Path(__file__).read_bytes()) == plan['script_sha256']
    assert sha(PARENT.read_bytes()) == plan['parent_plan_sha256']
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(20261114)
    tf.config.experimental.enable_op_determinism()
    if args.setting == 'no_constant_folding_or_arithmetic':
        tf.config.optimizer.set_experimental_options(
            {'constant_folding': False, 'arithmetic_optimization': False})
    else:
        assert args.setting == 'default'
    raw = {}
    for path, expected in plan['source_sha256'].items():
        data = tracked(plan['source_gpu_ref'], path)
        assert sha(data) == expected
        raw[path] = data
    origin = json.loads(raw[PREFIX + 'results/4090-hey-broad-equalized-screen-execution-v1.json'])
    verified = {}
    for name, expected in origin['files'].items():
        marker = '.local/wakeword/microWakeWord/'
        if marker in name and name.endswith('.py'):
            relative = name.split(marker, 1)[1]
            assert sha((args.upstream / relative).read_bytes()) == expected
            verified[relative] = expected
    assert len(verified) == 20
    for name, expected in origin['versions'].items():
        assert importlib.metadata.version(name) == expected
    sys.path.insert(0, str(args.upstream))
    from microwakeword import mixednet
    from microwakeword.layers.stream import Stream

    architecture = ast.parse(raw[PREFIX + 'experiments/probe_hey_context_transfer_shapes_v1.py'])
    nodes = [node for node in architecture.body if
             (isinstance(node, ast.FunctionDef) and node.name == 'create') or
             (isinstance(node, ast.Assign) and any(
                 isinstance(target, ast.Name) and target.id == 'ARCH' for target in node.targets))]
    assert len(nodes) == 2
    namespace = dict(argparse=argparse, mixednet=mixednet, np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 '<bound-architecture>', 'exec'), namespace)
    primitive = tracked(plan['primitive_gpu_ref'], PREFIX + PRIMITIVE)
    assert sha(primitive) == plan['primitive_sha256']
    primitive_namespace = {}
    exec(compile(primitive, '<bound-affine-primitive>', 'exec'), primitive_namespace)
    affine = primitive_namespace['affine_ste']
    compile_plan = json.loads(raw[PREFIX + 'results/4090-hey-broad-consistency-fixed-grid-compile-plan-v1.json'])
    args.output_dir.mkdir(exist_ok=False)
    results = []
    for arm in plan['arms']:
        weights_path = args.payload_root / arm['weights_path']
        model_path = args.payload_root / arm['model_path']
        assert sha(weights_path.read_bytes()) == arm['weights_sha256']
        assert sha(model_path.read_bytes()) == arm['model_sha256']
        assert model_path.stat().st_size == arm['model_bytes']
        model = namespace['create'](225)
        model.load_weights(weights_path)
        before = [variable.numpy().copy() for variable in model.variables]
        streams = [layer for layer in model.layers if isinstance(layer, Stream)]
        depths = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.DepthwiseConv2D)]
        points = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        norms = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
        head, = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
        assert len(streams) == 6 and len(depths) == len(points) == len(norms) == 4

        def parameters():
            values = [streams[0].cell.kernel, tf.zeros([30], tf.float32)]
            for depth, point, norm in zip(depths, points, norms, strict=True):
                values.extend([depth.kernel, depth.bias])
                factor = norm.gamma * tf.math.rsqrt(norm.moving_variance + norm.epsilon)
                values.extend([point.kernel * factor[None, None, None, :],
                               norm.beta - norm.moving_mean * factor])
            values.extend([head.kernel * tf.constant(3/8, tf.float32),
                           head.bias * tf.constant(3/8, tf.float32) + 3])
            return values

        template, = [entry['template'] for entry in compile_plan['arms'] if
                     (entry['target'], entry['condition']) == (arm['target'], arm['condition'])]
        ops = [op for op in template['ops'] if op['op'] in
               ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED')]
        assert len(ops) == 10
        interpreter = Interpreter(model_content=model_path.read_bytes(),
                                  experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                                  num_threads=1)
        interpreter.allocate_tensors()
        details = {detail['index']: detail for detail in interpreter.get_tensor_details()}
        grids = []
        for op in ops:
            for ordinal, index in enumerate(op['inputs'][1:]):
                detail = details[index]
                scales = detail['quantization_parameters']['scales']
                zero = detail['quantization_parameters']['zero_points'].astype(np.float32)
                if ordinal == 0:
                    if op['op'] == 'DEPTHWISE_CONV_2D':
                        scales, zero = scales.reshape(1, 1, -1, 1), zero.reshape(1, 1, -1, 1)
                    elif op['op'] == 'CONV_2D':
                        scales, zero = scales.reshape(1, 1, 1, -1), zero.reshape(1, 1, 1, -1)
                    else:
                        scales, zero = scales.reshape(1, -1), zero.reshape(1, -1)
                grids.append(dict(index=index, kind=op['op'], bias=bool(ordinal),
                                  scale=scales.copy(), zero=zero.copy(),
                                  actual=interpreter.get_tensor(index).copy()))
        assert len(grids) == 20

        def evaluate(runtime_scales=None):
            values = parameters()
            outputs = []
            for n, (value, grid) in enumerate(zip(values, grids, strict=True)):
                scale = grid['scale'] if runtime_scales is None else runtime_scales[n]
                bias = grid['bias']
                low, high = (-(2**31), 2**31-1) if bias else (-127, 127)
                dequant = affine(value, scale, grid['zero'], low, high, tf, wide=bias)
                precision = tf.float64 if bias else tf.float32
                quotient = tf.cast(value, precision) / tf.cast(scale, precision)
                outputs.extend([tf.identity(value), dequant, quotient])
            return tuple(outputs)

        captured = tf.function(evaluate, input_signature=[])
        signature = [tf.TensorSpec(grid['scale'].shape, tf.float32) for grid in grids]
        runtime = tf.function(lambda *scales: evaluate(scales), input_signature=signature)
        modes = {
            'eager_captured_scale': lambda: evaluate(),
            'graph_captured_scale': lambda: captured(),
            'graph_runtime_scale': lambda: runtime(*[tf.constant(g['scale']) for g in grids])}
        arrays, observations = {}, []
        for method, invoke in modes.items():
            repeats = [[value.numpy().copy() for value in invoke()] for _ in range(2)]
            assert len(repeats[0]) == 60
            assert all(np.isfinite(value).all() and np.array_equal(value, other)
                       for value, other in zip(*repeats, strict=True))
            for n, grid in enumerate(grids):
                value, dequant, quotient = repeats[0][n*3:n*3+3]
                scale = grid['scale'].astype(np.float64)
                codes = np.rint(dequant.astype(np.float64)/scale + grid['zero']).astype(np.int64)
                if not grid['bias']:
                    codes = codes.T if grid['kind'] == 'FULLY_CONNECTED' else np.transpose(codes, (3, 0, 1, 2))
                actual = grid['actual']
                assert codes.shape == actual.shape
                indices = np.argwhere(codes != actual)
                key = method + '_tensor' + str(grid['index'])
                for label, data in [('value', value), ('dequant', dequant), ('quotient', quotient),
                                    ('codes', codes), ('compiled', actual), ('scale', grid['scale'])]:
                    arrays[key + '_' + label] = data
                observations.append(dict(method=method, index=grid['index'], bias=grid['bias'],
                    values=int(codes.size), unequal=len(indices),
                    differences=[dict(index=index.tolist(), observed=int(codes[tuple(index)]),
                                      compiled=int(actual[tuple(index)])) for index in indices]))
        unchanged = all(np.array_equal(snapshot, variable.numpy())
                        for snapshot, variable in zip(before, model.variables, strict=True))
        assert unchanged
        destination = args.output_dir / (arm['target'] + '-' + arm['condition'] + '.npz')
        np.savez_compressed(destination, **arrays)
        entry = dict(target=arm['target'], condition=arm['condition'],
                     model_sha256=arm['model_sha256'], final_h5_sha256=arm['weights_sha256'],
                     observations=observations, repeats_exact=True, variables_unchanged=True,
                     arrays=destination.name, arrays_sha256=sha(destination.read_bytes()))
        results.append(entry)
        print(json.dumps(dict(target=arm['target'], condition=arm['condition'],
              unequal={method: sum(row['unequal'] for row in observations if row['method'] == method)
                       for method in modes})), flush=True)
    report = dict(schema='mac-all-final-constant-capture-result-v1',
                  plan_sha256=sha(PLAN.read_bytes()), setting=args.setting,
                  optimizer_options=tf.config.optimizer.get_experimental_options(),
                  physical_gpus_visible=tf.config.get_visible_devices('GPU'),
                  cases=results, constant_expressions_per_method=80,
                  model_invocations=0, optimizer_updates=0,
                  upstream_python_sha256=verified, versions=origin['versions'],
                  limits=plan['policy'])
    with (args.output_dir / 'report.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'worker'])
    parser.add_argument('--gpu-ref')
    parser.add_argument('--setting', choices=['default', 'no_constant_folding_or_arithmetic'])
    parser.add_argument('--payload-root', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    make_plan(args) if args.action == 'plan' else worker(args)
