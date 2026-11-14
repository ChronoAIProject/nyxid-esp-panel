"""Constructed semantics of unchanged raw-logit consistency and bounded head.

No learned model/data, optimizer, new development, audio or board access.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import tensorflow as tf


ROOT = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run(a):
    ref = subprocess.check_output(['git', 'rev-parse', 'f2578cf'], text=True).strip()
    paths = ['experiments/hey_broad_consistency_loss_v1.py',
             'experiments/probe_hey_broad_bounded_float_v1.py',
             'experiments/audit_hey_broad_batches_v1.py',
             'experiments/run_hey_broad_consistency_v1.py',
             'results/4090-hey-broad-consistency-formal-execution-v1.json',
             'results/4090-hey-broad-consistency-hey_willow-dual_supervised_control-training-original-v1.json']
    raw = {p: subprocess.check_output(['git', 'show', ref + ':' + ROOT + p]) for p in paths}
    execution = json.loads(raw[paths[-2]])
    for package in ['tensorflow', 'numpy']:
        assert importlib.metadata.version(package) == execution['versions'][package]
    for name in paths[:4]:
        expected, = [value for key, value in execution['files'].items()
                     if key == ROOT + name or key.endswith('/' + ROOT + name)]
        assert digest(raw[name]) == expected
    tf.config.set_visible_devices([], 'GPU')
    loss = {}
    exec(compile(raw[paths[0]], '<unchanged-raw-consistency>', 'exec'), loss)
    tree = ast.parse(raw[paths[1]])
    bounded, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'bounded']
    namespace = dict(np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[bounded], type_ignores=[])),
                 '<unchanged-bounded-helper>', 'exec'), namespace)
    inputs = tf.keras.Input(shape=(1,))
    dense = tf.keras.layers.Dense(1)
    model = tf.keras.Model(inputs, dense(inputs))
    dense.set_weights([np.ones((1, 1), np.float32), np.zeros(1, np.float32)])
    clipped = namespace['bounded'](model, tf)
    before = [v.numpy().copy() for v in clipped.variables]
    cases = []
    for zf, zq, label in [(-64., -128., 0.), (64., 128., 1.),
                          (-8., -9., 0.), (8., 9., 1.),
                          (1., -1., 1.), (.5, -.5, 1.)]:
        left = tf.Variable([[zf]], dtype=tf.float32)
        right = tf.Variable([[zq]], dtype=tf.float32)
        with tf.GradientTape() as tape:
            total, supervised, consistency = loss['components'](
                left, right, tf.constant([[label]], tf.float32),
                tf.constant([20. if label == 0 else 1.], tf.float32), 1, tf)
        gradients = tape.gradient(consistency, [left, right])
        probabilities = clipped(np.asarray([[zf], [zq]], np.float32), training=False).numpy().reshape(-1)
        delta = float(abs(float(probabilities[0]) - float(probabilities[1])))
        cases.append(dict(float_logit=zf, quant_surrogate_logit=zq,
            supervised_loss=float(supervised), consistency_loss=float(consistency),
            consistency_logit_gradients=[float(g.numpy()[0, 0]) for g in gradients],
            constructed_bounded_probabilities=[float(value) for value in probabilities],
            constructed_bounded_probability_abs_difference=delta))
    assert all(np.array_equal(value, variable.numpy()) for value, variable in zip(before, clipped.variables, strict=True))
    assert all(row['consistency_loss'] > 0 and row['constructed_bounded_probability_abs_difference'] == 0 for row in cases[:4])
    assert cases[0]['consistency_loss'] == 7.5 and cases[4]['consistency_loss'] == .03125
    assert cases[4]['constructed_bounded_probability_abs_difference'] > .46
    # Conditional scalar bound from the declared 128-example policy with
    # weights >=1: each individual branch BCE <=2*N*S. Factor2 gives a wide
    # numerical margin for the diagnostic, and is not an admission tolerance.
    history = json.loads(raw[paths[-1]])['history']
    boundary_bce = math.log1p(math.exp(-8))
    observations = []
    for row in history:
        upper = 2 * 128 * row['supervised_loss'] * 2
        observations.append(dict(step=row['step'], reported_supervised_loss=row['supervised_loss'],
            reported_consistency_loss=row['consistency_loss'],
            conservative_individual_bce_upper_under_declared_policy=upper,
            upper_below_bce_at_abs8=upper < boundary_bce))
    assert observations[0]['upper_below_bce_at_abs8'] and observations[-1]['upper_below_bce_at_abs8']
    report = dict(schema='mac-raw-consistency-saturation-semantics-v1', gpu_ref=ref,
        script_sha256=digest(Path(__file__).read_bytes()),
        source_sha256={ROOT + path: digest(data) for path, data in raw.items()},
        constructed_cases=cases,
        saturated_vs_boundary_consistency_penalty_ratio=cases[0]['consistency_loss'] / cases[4]['consistency_loss'],
        unchanged_bounded_toy_variables=True, optimizer_updates=0,
        conditional_logged_batch_observations=observations,
        bce_at_abs8=boundary_bce,
        all_constructed_checks_passed=True,
        limits='Actual unchanged GPU components() and bounded() run on a constructed identity Dense head. Both logits use the nominal float bounded map only; this is not exact deployed quantized FC/MUL/ADD. Shows raw-logit consistency can rank saturated equal-probability pairs above large probability disagreements. Conditional log bound assumes declared128, binary labels and weights1/20; private actual logits/weights were not opened. Does not prove why current540 failed, loss contribution fractions in actual data, recognition change, or efficacy of another loss. No coefficient/checkpoint/gate changes, learned inference, training, fresh-development/acceptance, audio or P4 operations.')
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_sha256', 'conditional_logged_batch_observations', 'constructed_cases']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
