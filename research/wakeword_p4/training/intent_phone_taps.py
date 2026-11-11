"""Training-only local/deep intent-phone fit with identical right endpoints.

This is a fresh representation experiment, not an extension of the rejected
phone fit, a pronunciation annotation, or an open-set wake detector.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from intent_phone_fit import (targets, hypothesis_scores, unique_bank,
                              update_factory)
from phone_sequences import collapse, edit_distance
from phrase_ctc_readout import load_bank, sha, summary, weights_sha


def encoder_tap(binary, kind):
    """Return 67 states at feature endpoints 136+3*j; no future features."""
    import tensorflow as tf
    if kind == 'deep':
        value = tf.keras.layers.Reshape((67, 60))(binary.get_layer('classifier').input)
    elif kind == 'local':
        # Post-activation of pointwise_1, found structurally rather than by
        # Keras's process-global auto-generated ReLU layer name.
        layer = binary.get_layer('pointwise_1')
        index = binary.layers.index(layer)
        activation = binary.layers[index+1]
        if not isinstance(activation, tf.keras.layers.ReLU) or tuple(activation.output.shape[1:]) != (99, 1, 60):
            raise ValueError('Unexpected shallow activation topology')
        cropped = tf.keras.layers.Cropping2D(((32, 0), (0, 0)))(activation.output)
        value = tf.keras.layers.Reshape((67, 60))(cropped)
    else:
        raise ValueError('Unknown encoder tap')
    if tuple(value.shape[1:]) != (67, 60):
        raise ValueError('Mismatched tap geometry')
    return tf.keras.Model(binary.input, value)


def fits(metrics, gate):
    return (metrics['macro_accuracy'] >= gate['minimum_macro_accuracy']
        and metrics['nyx_recall'] >= gate['minimum_nyx_recall']
        and metrics['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
        and metrics['greedy_exact_sequences']/metrics['rows'] >= gate['minimum_greedy_exact_sequence_fraction']
        and metrics['mean_ctc_nll_per_token'] <= gate['maximum_mean_normalized_ctc_loss'])


def phone_head(initialization, seed):
    """Equal-width phone head with explicit, auditable initialization policy."""
    import tensorflow as tf
    if initialization == 'glorot':
        kernel = tf.keras.initializers.GlorotUniform(seed=seed)
    elif initialization == 'zero':
        kernel = tf.keras.initializers.Zeros()
    else:
        raise ValueError('Unknown phone-head initialization')
    h = tf.keras.Input((67, 60))
    return tf.keras.Model(h, tf.keras.layers.Dense(10, kernel_initializer=kernel)(h))


def fit(kind, x, labels, plan, initial, output, *, tap=None, head_initialization='glorot',
        conditional_gain=None):
    import tensorflow as tf
    from transfer import build_model
    tf.keras.utils.set_random_seed(726)
    binary, _ = build_model(335)
    binary.load_weights(initial)
    binary_before = weights_sha(binary.weights)
    encoder = encoder_tap(binary, tap if tap is not None else kind)
    encoder_before = weights_sha(encoder.weights)
    trainable_ids = {id(v) for v in encoder.trainable_variables}
    disconnected = [v for v in binary.weights if id(v) not in trainable_ids]
    disconnected_before = weights_sha(disconnected)
    head = phone_head(head_initialization, plan['head_seed'])
    head_before = weights_sha(head.weights)
    update_plan = plan if conditional_gain is None else {**plan, 'conditional_ctc_gain': conditional_gain}
    full, update = update_factory(encoder, head, update_plan)
    packed, lengths = targets(labels, 'phones')
    initial_diagnostic = None
    if plan.get('initial_diagnostic_crop_ids'):
        diagnostic_ids = np.asarray(plan['initial_diagnostic_crop_ids'], np.int32)
        states = np.asarray(encoder(x[diagnostic_ids], training=False))
        values = np.asarray(head(states, training=False))
        probabilities = np.asarray(tf.nn.softmax(values, axis=-1))
        initial_array = output/f'{kind}-initial.npz'
        np.savez_compressed(initial_array, ids=diagnostic_ids, states=states, logits=values)
        initial_diagnostic = {'array': initial_array.name, 'array_sha256': sha(initial_array),
            'minimum_logit': float(values.min()), 'maximum_logit': float(values.max()),
            'mean_entropy': float(-np.mean(np.sum(probabilities*np.log(np.maximum(probabilities, 1e-30)), axis=-1))),
            'mean_maximum_probability': float(probabilities.max(-1).mean()),
            'argmax_counts': np.bincount(values.argmax(-1).ravel(), minlength=10).tolist()}
    ids = np.arange(len(x), dtype=np.int32)
    rng = np.random.default_rng(plan['draw_seed'])
    digest, history, started = hashlib.sha256(), [], time.monotonic()
    for step in range(1, plan['steps']+1):
        selected = rng.choice(ids, size=plan['batch_size'], replace=True).astype(np.int32)
        digest.update(selected.astype('<i4', copy=False).tobytes())
        if conditional_gain:
            loss, norm = update(x[selected], packed[selected], lengths[selected], labels[selected])
        else:
            loss, norm = update(x[selected], packed[selected], lengths[selected])
        if step % 250 == 0 or step == plan['steps']:
            item = {'step': step, 'loss': float(loss), 'gradient_norm_before_clip': float(norm)}
            history.append(item)
            print(json.dumps({'tap': kind, **item}), flush=True)
    parts, score_parts, total_loss = [], [], 0.
    for start in range(0, len(x), plan['batch_size']):
        selected = ids[start:start+plan['batch_size']]
        values = np.asarray(full(x[selected], training=False))
        likelihoods = hypothesis_scores(values, 'phones')
        parts.append(values)
        score_parts.append(likelihoods)
        total_loss += float(np.sum(-likelihoods[np.arange(len(selected)), labels[selected]]/lengths[selected]))
    logits, scores = np.concatenate(parts), np.concatenate(score_parts)
    decoded = [collapse(row) for row in logits.argmax(-1)]
    measured = summary(labels, scores.argmax(-1))
    measured.update(mean_ctc_nll_per_token=total_loss/len(ids),
        greedy_exact_sequences=sum(d == packed[i, :lengths[i]].tolist() for i, d in enumerate(decoded)),
        target_tokens=int(lengths.sum()),
        greedy_edits=sum(edit_distance(packed[i, :lengths[i]].tolist(), d) for i, d in enumerate(decoded)))
    measured['greedy_token_error_rate'] = measured['greedy_edits']/measured['target_tokens']
    if weights_sha(disconnected) != disconnected_before or weights_sha(encoder.weights) == encoder_before:
        raise ValueError('Wrong trainable scope or disconnected weights changed')
    checkpoint = output/f'{kind}.weights.h5'
    full.save_weights(checkpoint)
    arrays = output/f'{kind}-logits.npz'
    np.savez_compressed(arrays, logits=logits)
    result = {'summary': measured, 'parameters': full.count_params(), 'head_parameters': head.count_params(),
        'encoder_parameters': encoder.count_params(), 'history': history, 'scores': scores.tolist(),
        'decoded': decoded, 'draw_schedule_sha256': digest.hexdigest(),
        'binary_weights_before_sha256': binary_before,
        'encoder_weights_before_sha256': encoder_before,
        'encoder_weights_after_sha256': weights_sha(encoder.weights),
        'head_weights_before_sha256': head_before, 'head_weights_after_sha256': weights_sha(head.weights),
        'disconnected_weights_before_sha256': disconnected_before,
        'disconnected_weights_unchanged': True,
        'checkpoint': checkpoint.name, 'checkpoint_sha256': sha(checkpoint),
        'logits': arrays.name, 'logits_sha256': sha(arrays),
        'fit_gate_passed': fits(measured, plan['fit_gate']), 'seconds': time.monotonic()-started}
    if initial_diagnostic is not None:
        result['initial_diagnostic'] = initial_diagnostic
    if conditional_gain is not None:
        values = scores.astype(np.float64)
        shifted = values-values.max(-1, keepdims=True)
        result['conditional_objective'] = {'gain': conditional_gain,
            'mean_intent_nll': float(np.mean(np.log(np.exp(shifted).sum(-1))-shifted[ids, labels]))}
    (output/f'{kind}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'tap': kind, 'complete': True, **measured}), flush=True)
    tf.keras.backend.clear_session()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'initial-weights', 'parent-study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh study directory')
    plan = json.loads(args.plan.read_bytes())
    for name, digest in plan['source_sha256'].items():
        if sha(Path(__file__).parent/name) != digest:
            raise ValueError('Source differs: '+name)
    for name, digest in plan['parent_study_sha256'].items():
        if sha(args.parent_study/name) != digest:
            raise ValueError('Parent evidence differs: '+name)
    if (sha(args.initial_weights) != plan['initializer_sha256']
            or sha(args.features/'features.bin') != plan['feature_data_sha256']):
        raise ValueError('Initializer or actual scalar feature bytes differ')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    raw, labels, rows = load_bank(args.bank, args.features, plan['compact_bank'])
    raw, labels, rows = unique_bank(raw, labels, rows)
    if (rows != json.loads((args.parent_study/'study/rows.json').read_bytes())
            or len(rows) != plan['unique_crops']):
        raise ValueError('Training crop inventory differs')
    x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    results = {kind: fit(kind, x, labels, plan, args.initial_weights, args.output)
               for kind in ('deep', 'local')}
    deep, local = results['deep'], results['local']
    parent = json.loads((args.parent_study/'study/full-fit-phones.json').read_bytes())
    if any(deep[a] != parent[b] for a, b in (
            ('encoder_weights_after_sha256', 'encoder_weights_after_sha256'),
            ('head_weights_after_sha256', 'head_weights_sha256'),
            ('draw_schedule_sha256', 'draw_schedule_sha256'), ('summary', 'summary'))):
        raise ValueError('Deep control does not replay the prior full-fit tensors and metrics')
    if any(deep[k] != local[k] for k in ('binary_weights_before_sha256',
            'head_weights_before_sha256', 'draw_schedule_sha256', 'head_parameters')):
        raise ValueError('Paired initialization, head size or draws differ')
    report = {'plan_sha256': sha(args.plan), 'rows_sha256': sha(args.output/'rows.json'),
        'parent_deep_control_replayed': True,
        'arms': {kind: {'result': f'{kind}.json', 'result_sha256': sha(args.output/f'{kind}.json'),
                        'summary': value['summary'], 'fit_gate_passed': value['fit_gate_passed'],
                        'parameters': value['parameters']} for kind, value in results.items()},
        'advance': local['fit_gate_passed'], 'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, **report}), flush=True)


if __name__ == '__main__':
    main()
