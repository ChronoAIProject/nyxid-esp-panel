"""Training-only compact phrase CTC with a trainable temporal encoder.

This separate experiment tests representation learning after the failed frozen
readout. It neither extends that readout budget nor evaluates a wake detector.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from phrase_ctc_readout import (TOKENS, fit as frozen_fit, hypothesis_scores,
    load_bank, passes, sha, sha_array, summary, targets, weights_sha)


def make_head(shape, seed):
    import tensorflow as tf
    inputs = tf.keras.Input(shape)
    logits = tf.keras.layers.Dense(len(TOKENS),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(inputs)
    return tf.keras.Model(inputs, logits)


def make_update(encoder, head, plan):
    """One shared gradient clip, then independent fixed-rate Adam optimizers."""
    import tensorflow as tf
    full = tf.keras.Model(encoder.input, head(encoder.output))
    encoder_vars, head_vars = encoder.trainable_variables, head.trainable_variables
    if {id(v) for v in encoder_vars} & {id(v) for v in head_vars}:
        raise ValueError('Optimizer variable groups overlap')
    encoder_optimizer = tf.keras.optimizers.Adam(plan['encoder_learning_rate'])
    head_optimizer = tf.keras.optimizers.Adam(plan['learning_rate'])

    @tf.function
    def update(x, labels):
        with tf.GradientTape() as tape:
            logits = full(x, training=True)
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, logits=logits,
                label_length=tf.fill([tf.shape(x)[0]], 2),
                logit_length=tf.fill([tf.shape(x)[0]], encoder.output_shape[1]),
                logits_time_major=False, blank_index=0))/2
            tf.debugging.assert_all_finite(loss, 'Nonfinite CTC loss')
        variables = encoder_vars+head_vars
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise ValueError('Disconnected trainable variable')
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, 'Nonfinite CTC gradient')
        clipped, norm = tf.clip_by_global_norm(gradients, 5.)
        split = len(encoder_vars)
        encoder_optimizer.apply_gradients(zip(clipped[:split], encoder_vars, strict=True))
        head_optimizer.apply_gradients(zip(clipped[split:], head_vars, strict=True))
        return loss, norm

    return full, update


def measure(full, features, labels, batch_size):
    from phone_sequences import collapse
    values = np.concatenate([np.asarray(full(features[start:start+batch_size], training=False))
                             for start in range(0, len(features), batch_size)])
    scores = hypothesis_scores(values)
    measured = summary(labels, scores.argmax(-1))
    expected = targets(labels).tolist()
    decoded = [collapse(row) for row in values.argmax(-1)]
    measured.update(greedy_exact_sequence_fraction=float(np.mean([
        a == b for a, b in zip(decoded, expected, strict=True)])),
        mean_normalized_ctc_loss=float(-scores[np.arange(len(labels)), labels].mean()/2))
    return measured, scores, decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh result directory')
    plan = json.loads(args.plan.read_bytes())
    if (any(sha(Path(__file__).parent/name) != value for name, value in plan['source_sha256'].items())
            or sha(args.initial_weights) != plan['initializer_sha256']):
        raise ValueError('Executed source or initializer differs')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream source differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from transfer import build_model
    raw, labels, rows = load_bank(args.bank, args.features, plan)
    features = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    binary, _ = build_model(335)
    binary.load_weights(args.initial_weights)
    classifier = binary.get_layer('classifier')
    classifier_before = weights_sha(classifier.weights)
    temporal = tf.keras.layers.Reshape((67, 60))(classifier.input)
    encoder = tf.keras.Model(binary.input, temporal)
    encoder_before = weights_sha(encoder.weights)
    cache = np.concatenate([np.asarray(encoder(features[start:start+32], training=False))
                            for start in range(0, len(features), 32)])
    if sha_array(cache) != plan['required_encoder_cache_sha256']:
        raise ValueError('Original temporal representation is not reproduced')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    ids = np.arange(len(labels), dtype=np.int32)
    frozen, reference = frozen_fit(cache, labels, ids, ids, plan, 'ctc')
    if reference['head_weights_sha256'] != plan['required_frozen_head_sha256']:
        raise ValueError('Frozen readout replay differs')
    frozen.save_weights(args.output/'frozen-head.weights.h5')
    print(json.dumps({'stage': 'frozen_replay', **reference['summary']}), flush=True)
    tf.keras.utils.set_random_seed(plan['head_seed'])
    head = make_head(cache.shape[1:], plan['head_seed'])
    full, update = make_update(encoder, head, plan)
    before, _, _ = measure(full, features, labels, 32)
    rng = np.random.default_rng(plan['draw_seed'])
    import hashlib
    draws, history, started = hashlib.sha256(), [], time.monotonic()
    packed = tf.constant(targets(labels))
    x = tf.constant(features)
    for step in range(1, plan['steps']+1):
        selected = rng.choice(ids, size=plan['batch_size'], replace=True).astype(np.int32)
        draws.update(selected.astype('<i4', copy=False).tobytes())
        loss, norm = update(tf.gather(x, selected), tf.gather(packed, selected))
        if step % 250 == 0 or step == plan['steps']:
            row = {'step': step, 'loss': float(loss), 'gradient_norm_before_clip': float(norm),
                   'seconds': time.monotonic()-started}
            history.append(row)
            print(json.dumps({'stage': 'encoder_training', **row}), flush=True)
    measured, scores, decoded = measure(full, features, labels, 32)
    if draws.hexdigest() != reference['draw_schedule_sha256']:
        raise ValueError('Encoder and frozen-head row draws differ')
    if weights_sha(classifier.weights) != classifier_before:
        raise ValueError('Disconnected binary classifier changed')
    if weights_sha(encoder.weights) == encoder_before:
        raise ValueError('Temporal encoder did not train')
    encoder.save_weights(args.output/'encoder.weights.h5')
    head.save_weights(args.output/'trainable-head.weights.h5')
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan),
        'runtime': {'python': sys.version, 'packages': {name: importlib.metadata.version(name)
            for name in ('tensorflow', 'numpy', 'keras', 'h5py')}},
        'rows': rows, 'frozen_reference': reference,
        'encoder_cache_sha256': sha_array(cache),
        'encoder_weights_before_sha256': encoder_before,
        'encoder_weights_after_sha256': weights_sha(encoder.weights),
        'binary_classifier_unchanged': True, 'initial_joint_fit': before,
        'trainable': {'parameters': full.count_params(), 'initial_fit': before,
            'summary': measured, 'scores': scores.tolist(), 'decoded': decoded,
            'history': history, 'draw_schedule_sha256': draws.hexdigest(),
            'head_weights_sha256': weights_sha(head.weights)},
        'advance': passes(measured, plan['fit_gate']), 'accepted': False,
        'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'advance': report['advance'], **measured}), flush=True)


if __name__ == '__main__':
    main()
