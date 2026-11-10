"""Prepare matched mean-pool and ordered dense wake-head initializations.

Shared convolution/BatchNorm and sigmoid weights are identical. The added
Dense32 starts as the uniform temporal mean of the same eight positions, so
both models initially give the same function up to float accumulation order.
Only completed training representatives are used for this interface audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_context_head import make_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--representatives', type=Path, required=True)
    p.add_argument('--representatives-sha256', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.seed < 0:
        p.error('Fresh output and nonnegative seed required')
    if digest(args.source) != args.source_sha256 or digest(args.representatives) != args.representatives_sha256:
        raise ValueError('Explicit training inputs differ')
    tf.keras.utils.set_random_seed(args.seed)
    source = tf.keras.models.load_model(args.source)
    control = make_model('temporal32-bn-mean-dropout')
    control.set_weights(source.get_weights())
    treatment = make_model('temporal32-bn-dense32-dropout')
    for left, right in zip(control.layers[:7], treatment.layers[:7]):
        right.set_weights(left.get_weights())
    dense = [layer for layer in treatment.layers if isinstance(layer, tf.keras.layers.Dense)]
    dense[0].set_weights([np.tile(np.eye(32, dtype=np.float32) / 8, (8, 1)), np.zeros(32, np.float32)])
    dense[1].set_weights(control.layers[-1].get_weights())
    if not all(np.array_equal(a, b) for left, right in zip(control.layers[:7], treatment.layers[:7])
               for a, b in zip(left.get_weights(), right.get_weights())):
        raise ValueError('Common feature weights differ')
    if not all(np.array_equal(a, b) for a, b in zip(control.layers[-1].get_weights(), dense[1].get_weights())):
        raise ValueError('Common classifier weights differ')
    examples = np.load(args.representatives, allow_pickle=False)
    if examples.shape != (512, 16, 96, 1) or examples.dtype != np.float32 or not np.isfinite(examples).all():
        raise ValueError('Expected the existing 512 training-only representative histories')
    delta = float(np.abs(control(examples, training=False).numpy() - treatment(examples, training=False).numpy()).max())
    if delta >= 1e-5:
        raise ValueError('Initial functions differ beyond float accumulation tolerance')
    args.output.mkdir(parents=True)
    outputs = {}
    for arm, model in [('control', control), ('dense', treatment)]:
        destination = args.output / (arm + '.keras')
        model.save(destination)
        outputs[arm] = {'sha256': digest(destination), 'parameters': model.count_params()}
    report = {'scope': __doc__, 'source_sha256': args.source_sha256,
              'representatives_sha256': args.representatives_sha256, 'seed': args.seed,
              'shared_feature_arrays_exact': True, 'classifier_arrays_exact': True,
              'initial_float_max_delta': delta, 'fixtures': len(examples), 'outputs': outputs,
              'accepted': False}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
