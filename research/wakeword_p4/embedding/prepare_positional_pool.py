"""Prepare matched fixed/learnable temporal-pool heads from verified weights.

Use training representatives only. This verifies initial float equivalence;
it neither trains nor evaluates wake recognition or MCU compatibility.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_head import make_model


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--representatives', type=Path, required=True)
    p.add_argument('--representatives-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    for path, expected in ((args.source, args.source_sha256),
                           (args.representatives, args.representatives_sha256)):
        if digest(path) != expected:
            raise ValueError('Frozen initialization or training fixture identity differs')
    examples = np.load(args.representatives, allow_pickle=False)
    if examples.shape != (512, 16, 96, 1) or not np.isfinite(examples).all():
        raise ValueError('Expected the complete fixed training representative histories')
    source = tf.keras.models.load_model(args.source, compile=False)
    source_layers = [layer for layer in source.layers if layer.get_weights()]
    if [type(layer).__name__ for layer in source_layers] != [
            'Conv2D', 'BatchNormalization', 'Conv2D', 'BatchNormalization', 'Dense']:
        raise ValueError('Expected the original temporal head learned tensor layout')
    mean = make_model('temporal32-bn-mean-dropout')
    mean.set_weights(source.get_weights())
    reference_features = tf.keras.Model(mean.input, mean.layers[-1].input)(examples, training=False).numpy()
    reference = mean(examples, training=False).numpy()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'scope': __doc__, 'source_sha256': args.source_sha256,
              'representatives_sha256': args.representatives_sha256, 'arms': {},
              'acceptance': False}
    previous = None
    for arm in ('fixed', 'learned'):
        tf.keras.utils.set_random_seed(727)
        model = make_model(f'temporal32-bn-{arm}-pool-dropout')
        layers = [layer for layer in model.layers if layer.get_weights()
                  and not isinstance(layer, tf.keras.layers.DepthwiseConv2D)]
        if len(layers) != len(source_layers):
            raise ValueError('Learned tensor layout changed')
        for target, old in zip(layers, source_layers):
            target.set_weights(old.get_weights())
            if any(not np.array_equal(a, b) for a, b in zip(target.get_weights(), old.get_weights())):
                raise ValueError('Original learned tensors changed')
        pools = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.DepthwiseConv2D)]
        pool, = pools
        weights, = pool.get_weights()
        if weights.shape != (8, 1, 32, 1) or not (weights == .125).all():
            raise ValueError('Expected exactly uniform bias-free pooling weights')
        if pool.trainable != (arm == 'learned'):
            raise ValueError('Pool trainability differs from the arm')
        features = tf.keras.Model(model.input, model.layers[-1].input)(examples, training=False).numpy()
        prediction = model(examples, training=False).numpy()
        if not np.allclose(reference_features, features, rtol=0, atol=3e-6):
            raise ValueError('Average-pooling feature equivalence failed')
        if not np.allclose(reference, prediction, rtol=0, atol=3e-6):
            raise ValueError('Average-pooling probability equivalence failed')
        if previous is not None and not np.array_equal(previous, prediction):
            raise ValueError('Matched arm initial predictions differ')
        previous = prediction
        path = args.output / (arm + '.keras')
        model.save(path)
        report['arms'][arm] = {'initial_weights_sha256': digest(path),
            'original_learned_tensors_exact': True, 'pool_trainable': pool.trainable,
            'pool_values_exact': True,
            'max_feature_delta_from_mean': float(np.max(np.abs(features - reference_features))),
            'max_probability_delta_from_mean': float(np.max(np.abs(prediction - reference)))}
    report['matched_initial_predictions_exact'] = True
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
