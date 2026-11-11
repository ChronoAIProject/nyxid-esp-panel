"""Compare actual fitted-head CTC losses/gradients with independent NumPy math."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ctc_reference import loss_and_gradient
from intent_phone_fit import targets, unique_bank
from phrase_ctc_readout import load_bank, sha, weights_sha


def compare(values, labels, lengths):
    import tensorflow as tf
    variable = tf.Variable(values, dtype=tf.float32)

    @tf.function
    def losses(x):
        return tf.nn.ctc_loss(labels, x, lengths, [values.shape[1]]*len(values),
                             logits_time_major=False, blank_index=0)

    with tf.GradientTape() as tape:
        actual = losses(variable)
        objective = tf.reduce_sum(actual)
    actual_gradient = tape.gradient(objective, variable).numpy()
    references = [loss_and_gradient(x, row[:length].tolist())
                  for x, row, length in zip(values, labels, lengths, strict=True)]
    expected_loss = np.array([v[0] for v in references])
    expected_gradient = np.stack([v[1] for v in references])
    loss_delta = np.abs(np.asarray(actual, np.float64)-expected_loss)
    gradient_delta = np.abs(actual_gradient-expected_gradient)
    return {'tf_loss': actual.numpy(), 'reference_loss': expected_loss,
            'tf_gradient': actual_gradient, 'reference_gradient': expected_gradient,
            'summary': {'samples': len(values), 'minimum_logit': float(values.min()),
                'maximum_logit': float(values.max()),
                'maximum_loss_absolute_delta': float(loss_delta.max()),
                'maximum_gradient_absolute_delta': float(gradient_delta.max()),
                'gradient_relative_l2_delta': float(np.linalg.norm(actual_gradient-expected_gradient)
                                                   /max(np.linalg.norm(expected_gradient), 1e-30)),
                'zero_gradients_with_reference_magnitude_above_0_001': int(np.sum(
                    (actual_gradient == 0) & (np.abs(expected_gradient) > .001)))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'study', 'upstream', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh numerical diagnostic directory')
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan['source_sha256'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Numerical source differs: '+name)
    for name, digest in plan['study_files_sha256'].items():
        if sha(args.study/name) != digest:
            raise ValueError('Fitted study bytes differ: '+name)
    if (sha(args.initial_weights) != plan['initializer_sha256']
            or sha(args.features/'features.bin') != plan['feature_data_sha256']):
        raise ValueError('Initializer or actual feature bytes differ')
    if subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() != plan['upstream_commit']:
        raise ValueError('Upstream revision differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    from transfer import build_model
    raw, labels, rows = load_bank(args.bank, args.features, plan['compact_bank'])
    raw, labels, rows = unique_bank(raw, labels, rows)
    retained = json.loads((args.study/'study/rows.json').read_bytes())
    if rows != retained:
        raise ValueError('Reconstructed unique crop inventory differs')
    ids = plan['training_crop_ids']
    if any(type(i) is not int or not 0 <= i < len(rows) for i in ids):
        raise ValueError('Invalid frozen training crop IDs')
    x = np.clip(raw[ids].astype(np.float32)/25.6, 0, 26)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    reports = {}
    for kind in ('words', 'phones'):
        tf.keras.utils.set_random_seed(726)
        binary, _ = build_model(335)
        binary.load_weights(args.initial_weights)
        classifier = binary.get_layer('classifier')
        encoder = tf.keras.Model(binary.input, tf.keras.layers.Reshape((67, 60))(classifier.input))
        h = tf.keras.Input((67, 60))
        head = tf.keras.Model(h, tf.keras.layers.Dense(6 if kind == 'words' else 10)(h))
        full = tf.keras.Model(encoder.input, head(encoder.output))
        full.load_weights(args.study/f'study/full-fit-{kind}.weights.h5')
        original = json.loads((args.study/f'study/full-fit-{kind}.json').read_bytes())
        if (weights_sha(encoder.weights) != original['encoder_weights_after_sha256']
                or weights_sha(head.weights) != original['head_weights_sha256']):
            raise ValueError('Loaded checkpoint tensor hashes differ')
        values = np.concatenate([np.asarray(full(x[start:start+32], training=False))
                                 for start in range(0, len(x), 32)])
        packed, lengths = targets(labels[ids], kind)
        checked = compare(values, packed, lengths)
        path = args.output/f'{kind}-actual.npz'
        np.savez_compressed(path, logits=values, labels=packed, lengths=lengths,
                            **{k: v for k, v in checked.items() if k != 'summary'})
        reports[kind] = {**checked['summary'], 'array': path.name, 'array_sha256': sha(path),
            'loaded_encoder_tensor_sha256': weights_sha(encoder.weights),
            'loaded_head_tensor_sha256': weights_sha(head.weights)}
        print(json.dumps({'kind': kind, **reports[kind]}), flush=True)
        tf.keras.backend.clear_session()
    synthetic = {}
    packed, lengths = targets(np.array([0, 2]), 'phones')
    for scale in plan['synthetic_logit_scales']:
        values = (np.random.default_rng(plan['synthetic_seed']).normal(size=(2, 67, 10))*scale).astype(np.float32)
        checked = compare(values, packed, lengths)
        path = args.output/f'synthetic-scale-{scale}.npz'
        np.savez_compressed(path, logits=values, labels=packed, lengths=lengths,
                            **{k: v for k, v in checked.items() if k != 'summary'})
        synthetic[str(scale)] = {**checked['summary'], 'array': path.name, 'array_sha256': sha(path)}
    result = {'plan_sha256': sha(args.plan), 'actual': reports, 'synthetic': synthetic,
        'training_crop_ids': ids, 'selected_rows': [rows[i] for i in ids],
        'optimization_performed': False, 'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'actual': reports, 'synthetic': synthetic}), flush=True)


if __name__ == '__main__':
    main()
