"""Audit binary/phone encoder gradients on the existing training bank only.

This small paired-bank diagnostic is not the full 128-example training batch.
It changes no weights, optimizer, data labels, threshold or inference graph.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from phone_distillation import Sampler, supervisor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'features', 'phone-bank', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output path')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(a.upstream.resolve()))
    from transfer import build_model
    index = (a.features/'features.json').read_bytes()
    bank = Sampler(a.phone_bank, json.loads(index), hashlib.sha256(index).hexdigest(), 335)
    model, _ = build_model(335)
    train = supervisor(model, bank.targets, .01, 'conditional')
    train.auxiliary_model.load_weights(a.checkpoint)
    before = [v.numpy().copy() for v in model.weights]
    classifier = {id(v) for v in model.get_layer('classifier').trainable_variables}
    variables = [v for v in model.trainable_variables if id(v) not in classifier]
    results = []
    for start in range(0, len(bank.features), 16):
        stop = min(start+16, len(bank.features))
        x = np.clip(bank.features[start:stop].astype(np.float32)/25.6, 0, 26)
        y = (1-np.arange(start, stop)%2).astype(np.float32)[:, None]
        weights = np.where(y[:, 0] > 0, 1., 5.).astype(np.float32)
        with tf.GradientTape(persistent=True) as tape:
            probability, logits = train.auxiliary_model(x, training=True)
            binary = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, probability)*weights)
            phone = train.phone_loss(tf.constant(bank.targets[start:stop]), logits)
            joint = binary+.01*phone
        vectors = []
        for loss in (binary, phone, joint):
            gradients = tape.gradient(loss, variables, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            vectors.append(np.concatenate([np.asarray(g).ravel() for g in gradients]).astype(np.float64))
        del tape
        gb, gp, gj = vectors
        nb, np_ = float(np.linalg.norm(gb)), float(np.linalg.norm(gp))
        residual = float(np.max(np.abs(gj-(gb+.01*gp))))
        if not np.isfinite([nb, np_, residual]).all() or residual > 1e-5*max(1, nb, np_):
            raise ValueError('Nonfinite or inconsistent gradient decomposition')
        results.append({'start': start, 'examples': stop-start,
                        'binary_loss': float(binary), 'phone_loss': float(phone),
                        'binary_encoder_gradient_norm': nb, 'phone_encoder_gradient_norm': np_,
                        'phone_to_binary_norm_at_gain_0_01': .01*np_/nb if nb else None,
                        'gradient_cosine': float(np.dot(gb, gp)/(nb*np_)) if nb*np_ else None,
                        'decomposition_max_error': residual})
    for previous, current in zip(before, model.weights, strict=True):
        np.testing.assert_array_equal(previous, current.numpy())
    report = {'scope': __doc__, 'source_sha256': sha(Path(__file__)),
              'bank_sha256': bank.bank_sha256, 'checkpoint_sha256': sha(a.checkpoint),
              'examples': len(bank.features), 'binary_weights_unchanged': True,
              'batches': results,
              'limits': 'Training-only paired crops; these gradient ratios do not describe the full mixed-source training batches or prove that increasing the gain helps validation.'}
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'batches'}, indent=2))
    for key in ('phone_to_binary_norm_at_gain_0_01', 'gradient_cosine'):
        values = [r[key] for r in results if r[key] is not None]
        print(key, json.dumps({'min': min(values), 'median': float(np.median(values)), 'max': max(values)}))


if __name__ == '__main__':
    main()
