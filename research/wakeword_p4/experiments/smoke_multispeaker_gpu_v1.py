"""Two actual GPU updates per arm only; no formal fit or candidate scoring."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np

R = Path('research/wakeword_p4/results')
PLAN = R / '4090-multispeaker-gpu-smoke-plan-v1.json'
OUT = Path('.local/wakeword/multispeaker-gpu-smoke-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_plan():
    probe = R / '4090-multispeaker-batch-probe-v1.json'
    audit = R / '4090-multispeaker-batch-independent-audit-v1.json'
    init_path = R / '4090-multispeaker-context309-initialization-v1.json'
    p, a, initial = [json.loads(path.read_text()) for path in [probe, audit, init_path]]
    assert a['original_report_sha256'] == sha(probe)
    assert a['all_role_label_source_window_and_actual_tensor_hashes_match']
    paths = [Path(__file__), probe, audit, init_path, Path(p['records_path']),
             Path(initial['snapshot']['path']), Path(initial['variable_arrays']['path']),
             Path(initial['variable_inventory']['path']),
             Path(__file__).with_name('hey_broad_gpu_training_factory_v1.py'),
             Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py'),
             Path(__file__).resolve().parents[1] / 'training/gpu_python.py']
    paths += list(Path('.local/wakeword/microWakeWord/microwakeword').rglob('*.py'))
    batches = {}
    for row in map(json.loads, Path(p['records_path']).read_text().splitlines()):
        if 'actual_tensor_path' in row:
            assert sha(row['actual_tensor_path']) == row['actual_tensor_sha256']
            paths.append(Path(row['actual_tensor_path']))
            batches.setdefault(row['arm'], []).append(row)
    assert set(batches) == {'narrow', 'broad'} and all(len(rows) == 2 for rows in batches.values())
    plan = dict(schema='multispeaker-gpu-smoke-plan-v1', files={str(p): sha(p) for p in paths},
                versions={n: importlib.metadata.version(n) for n in ['tensorflow', 'keras', 'numpy']},
                initial=initial, batches=batches, steps=2, seed=726, expected_new_gib=.02,
                optimizer=dict(learning_rate=1e-5, beta_1=.9, beta_2=.999, epsilon=1e-7),
                policy='Exact actual saved CPU first2 tensors per arm, same preserved context309 initialization; frozen BN/all15 trainables, raw sigmoid BCE weighted mean1/20, fresh Adam32variables/zero slots. RTX4090 required, FP32 TF32disabled, deterministic ops. No gradient/logit cross-device equality experiment. Save actual preupdate variables/optimizer and fixedfinal2; no export, formal training, selection or holdout scoring authorized.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN), updates_per_arm=2)))


def run():
    assert subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, h in plan['files'].items():
        assert sha(path) == h, path
    for package, version in plan['versions'].items():
        assert importlib.metadata.version(package) == version
    subprocess.run(['/usr/bin/python3', 'research/wakeword_p4/tools/check_disk_budget.py',
                    '--expected-gib', str(plan['expected_new_gib'])], check=True)
    import hey_broad_gpu_training_factory_v1 as factory
    tf = factory.tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    OUT.mkdir(exist_ok=False)
    results = []
    for arm in ['narrow', 'broad']:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(plan['seed'])
        root = OUT / arm
        root.mkdir()
        model = factory.create(309)
        model.load_weights(plan['initial']['snapshot']['path'])
        for layer in model.layers:
            cell = getattr(layer, 'cell', layer)
            if isinstance(cell, tf.keras.layers.BatchNormalization):
                layer.trainable = cell.trainable = False
        assert len(model.trainable_variables) == 15
        inventory = plan['initial']['variable_inventory']['variables']
        for v, expected in zip(model.variables, inventory, strict=True):
            assert hashlib.sha256(v.numpy().tobytes()).hexdigest() == expected['sha256']
        np.savez(root / 'initial-model-variables.npz', **{f'v{i:03}': v.numpy() for i, v in enumerate(model.variables)})
        bn_before = [v.numpy().copy() for v in model.non_trainable_variables]
        optimizer = tf.keras.optimizers.Adam(**plan['optimizer'])
        optimizer.build(model.trainable_variables)
        assert len(optimizer.variables) == 32 and int(optimizer.iterations.numpy()) == 0
        assert all(np.all(v.numpy() == 0) for v in optimizer.variables[2:])
        np.savez(root / 'initial-optimizer.npz', **{f'v{i:03}': v.numpy() for i, v in enumerate(optimizer.variables)})
        head = model.layers[-1]
        assert head.activation.__name__ == 'sigmoid'
        embedding = tf.keras.Model(model.inputs, head.input)
        steps = []
        for request in plan['batches'][arm]:
            with np.load(request['actual_tensor_path'], allow_pickle=False) as saved:
                x, y, weights = [saved[key].copy() for key in ['x', 'y', 'weights']]
            assert hashlib.sha256(x.tobytes()).hexdigest() == request['x_sha256']
            with tf.device('/GPU:0'):
                tx, ty, tw = tf.constant(x), tf.constant(y), tf.constant(weights)
                with tf.GradientTape() as tape:
                    logits = tf.linalg.matmul(embedding(tx, training=False), head.kernel) + head.bias
                    loss = tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=ty, logits=logits), [-1]) * tw)
                gradients = tape.gradient(loss, model.trainable_variables)
                assert 'GPU:0' in logits.device and all(g is not None for g in gradients)
                tf.debugging.assert_all_finite(loss, 'loss')
                for g in gradients:
                    tf.debugging.assert_all_finite(g, 'gradient')
                optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
            assert all(np.isfinite(v.numpy()).all() for v in model.variables)
            assert all(np.array_equal(a, v.numpy()) for a, v in zip(bn_before, model.non_trainable_variables, strict=True))
            item = dict(step=request['step'], loss=float(loss.numpy()), x_sha256=request['x_sha256'],
                        logits_device=logits.device, optimizer_iteration=int(optimizer.iterations.numpy()),
                        all15_gradients_finite=True, frozen_bn_unchanged=True)
            steps.append(item)
            print(json.dumps(dict(arm=arm, **item)), flush=True)
        assert int(optimizer.iterations.numpy()) == 2
        model.save_weights(root / 'final.weights.h5')
        np.savez(root / 'final-model-variables.npz', **{f'v{i:03}': v.numpy() for i, v in enumerate(model.variables)})
        np.savez(root / 'final-optimizer.npz', **{f'v{i:03}': v.numpy() for i, v in enumerate(optimizer.variables)})
        result = dict(arm=arm, steps=steps, trainable_variables=15, optimizer_variables=32,
                      initial_matches_preserved_snapshot=True, files={p.name: sha(p) for p in root.iterdir()})
        (root / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
        results.append(result)
    report = dict(schema='multispeaker-gpu-smoke-v1', plan_sha256=sha(PLAN), device=factory.identity,
                  arms=results, all_update_checks_passed=True, formal_training_started=False,
                  exports_created=False, development_scored=False, acceptance_scored=False,
                  limits='Two disposable integration updates per arm only. Streaming export/INT8 runtime admission and formal execution pins still required; no efficacy or candidate qualification.')
    with (R / '4090-multispeaker-gpu-smoke-v1.json').open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(all_update_checks_passed=True, arms=2, updates_per_arm=2)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'run'])
    args = parser.parse_args()
    write_plan() if args.action == 'plan' else run()
