"""Training-source diagnostic: shared intent phones versus whole-word CTC.

No ASR output becomes a label. These dictionary-like synthesis-intent strings
are hypotheses, not independently verified pronunciations or phonetic times.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from phrase_ctc_readout import (CLASSES, TOKENS as WORD_TOKENS, load_bank,
                               sha, summary, targets as word_targets, weights_sha)
from phone_sequences import collapse, edit_distance

TOKENS = ('blank', 'HH', 'EY', 'N', 'IH', 'K', 'S', 'M', 'EH', 'T')
PHONE_STRINGS = (('HH', 'EY', 'N', 'IH', 'K', 'S'),
                 ('HH', 'EY', 'M', 'IH', 'K', 'S'),
                 ('HH', 'EY', 'N', 'EH', 'K', 'S', 'T'),
                 ('HH', 'EY', 'N', 'IH', 'K'))


def targets(classes, kind):
    classes = np.asarray(classes)
    if (classes.ndim != 1 or not np.issubdtype(classes.dtype, np.integer)
            or np.any((classes < 0) | (classes >= len(CLASSES)))):
        raise ValueError('Expected four intent classes')
    if kind == 'words':
        return word_targets(classes), np.full(len(classes), 2, np.int32)
    if kind != 'phones':
        raise ValueError('Unknown target representation')
    labels = np.zeros((len(classes), 7), np.int32)
    lengths = np.empty(len(classes), np.int32)
    for i, label in enumerate(classes):
        sequence = [TOKENS.index(phone) for phone in PHONE_STRINGS[label]]
        labels[i, :len(sequence)] = sequence
        lengths[i] = len(sequence)
    return labels, lengths


def ctc_loss(labels, lengths, logits):
    import tensorflow as tf
    return tf.nn.ctc_loss(labels, logits, lengths,
        tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1]),
        logits_time_major=False, blank_index=0)


def hypothesis_scores(logits, kind):
    """Raw CTC log likelihoods with equal class priors, no length correction."""
    import tensorflow as tf
    labels, lengths = targets(np.arange(len(CLASSES)), kind)
    count = tf.shape(logits)[0]
    losses = ctc_loss(tf.tile(labels, [count, 1]), tf.tile(lengths, [count]),
                      tf.repeat(logits, len(CLASSES), axis=0))
    return np.asarray(tf.reshape(-losses, [count, len(CLASSES)]))


def conditional_intent_losses(logits, intent_ids):
    """Four fixed phone-string likelihoods as a differentiable intent objective."""
    import tensorflow as tf
    labels, lengths = targets(np.arange(len(CLASSES)), 'phones')
    count = tf.shape(logits)[0]
    losses = ctc_loss(tf.tile(labels, [count, 1]), tf.tile(lengths, [count]),
                      tf.repeat(logits, len(CLASSES), axis=0))
    scores = tf.reshape(-losses, [count, len(CLASSES)])
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent_ids, logits=scores)


def update_factory(encoder, head, plan):
    import tensorflow as tf
    full = tf.keras.Model(encoder.input, head(encoder.output))
    encoder_variables, head_variables = encoder.trainable_variables, head.trainable_variables
    if {id(v) for v in encoder_variables} & {id(v) for v in head_variables}:
        raise ValueError('Optimizer groups overlap')
    encoder_adam = tf.keras.optimizers.Adam(plan['encoder_learning_rate'])
    head_adam = tf.keras.optimizers.Adam(plan['head_learning_rate'])
    conditional_gain = float(plan.get('conditional_ctc_gain', 0))
    if not np.isfinite(conditional_gain) or conditional_gain < 0:
        raise ValueError('Expected a finite nonnegative conditional CTC gain')

    @tf.function
    def compiled_update(x, labels, lengths, intent_ids=None):
        with tf.GradientTape() as tape:
            logits = full(x, training=True)
            losses = ctc_loss(labels, lengths, logits)
            loss = tf.reduce_mean(losses/tf.cast(lengths, losses.dtype))
            if conditional_gain:
                loss += conditional_gain*tf.reduce_mean(conditional_intent_losses(logits, intent_ids))
            tf.debugging.assert_all_finite(loss, 'Nonfinite intent CTC loss')
        variables = encoder_variables+head_variables
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise ValueError('Disconnected optimizer variable')
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, 'Nonfinite intent CTC gradient')
        clipped, norm = tf.clip_by_global_norm(gradients, plan['global_clipnorm'])
        boundary = len(encoder_variables)
        encoder_adam.apply_gradients(zip(clipped[:boundary], encoder_variables, strict=True))
        head_adam.apply_gradients(zip(clipped[boundary:], head_variables, strict=True))
        return loss, norm

    def update(x, labels, lengths, intent_ids=None):
        # Validate before tracing: an invalid first tf.function call must not
        # partially initialize/cache Adam and break the subsequent valid call.
        if conditional_gain and intent_ids is None:
            raise ValueError('Conditional CTC requires the original intent classes')
        return compiled_update(x, labels, lengths, intent_ids)

    return full, update


def unique_bank(features, labels, rows):
    """Remove exact Cartesian crop replication, retaining identity consistency."""
    by_digest, chosen, metadata = {}, [], []
    for i, (value, label, row) in enumerate(zip(features, labels, rows, strict=True)):
        digest = hashlib.sha256(value.astype('<u2', copy=False).tobytes()).hexdigest()
        identity = (row['engine'], row['sha256'], row['variant'], int(label))
        if digest in by_digest:
            if by_digest[digest][0] != identity:
                raise ValueError('An identical crop has conflicting identity or intent')
            metadata[by_digest[digest][1]]['bank_rows'].append(i)
        else:
            by_digest[digest] = (identity, len(chosen))
            chosen.append(i)
            metadata.append({**row, 'class': int(label), 'crop_sha256': digest, 'bank_rows': [i]})
    return features[chosen], labels[chosen], metadata


def fit(kind, fold, x, labels, rows, train_ids, test_ids, plan, initial, output):
    import tensorflow as tf
    from transfer import build_model
    tf.keras.utils.set_random_seed(726)
    binary, _ = build_model(335)
    binary.load_weights(initial)
    classifier = binary.get_layer('classifier')
    classifier_hash = weights_sha(classifier.weights)
    temporal = tf.keras.layers.Reshape((67, 60))(classifier.input)
    encoder = tf.keras.Model(binary.input, temporal)
    encoder_before = weights_sha(encoder.weights)
    inputs = tf.keras.Input((67, 60))
    head = tf.keras.Model(inputs, tf.keras.layers.Dense(len(WORD_TOKENS) if kind == 'words' else len(TOKENS),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=plan['head_seed']))(inputs))
    full, update = update_factory(encoder, head, plan)
    packed, lengths = targets(labels, kind)
    probabilities = None
    if plan.get('intent_sampling', 'uniform-crop') != 'uniform-crop':
        from intent_sampling import hierarchical_probabilities
        if plan['intent_sampling'] != 'engine-class-source-render-crop':
            raise ValueError('Unknown intent sampling policy')
        probabilities = hierarchical_probabilities(rows, train_ids)
    rng = np.random.default_rng(plan['draw_seed'])
    schedule, history, started = hashlib.sha256(), [], time.monotonic()
    for step in range(1, plan['steps']+1):
        ids = rng.choice(train_ids, size=plan['batch_size'], replace=True,
                         p=probabilities).astype(np.int32)
        schedule.update(ids.astype('<i4', copy=False).tobytes())
        loss, norm = update(x[ids], packed[ids], lengths[ids])
        if step % 250 == 0 or step == plan['steps']:
            item = {'step': step, 'loss': float(loss), 'gradient_norm_before_clip': float(norm)}
            history.append(item)
            print(json.dumps({'fold': fold, 'kind': kind, **item}), flush=True)
    scores, decoded, total_loss, edits, target_count = [], [], 0., 0, 0
    for start in range(0, len(test_ids), plan['batch_size']):
        ids = test_ids[start:start+plan['batch_size']]
        logits = np.asarray(full(x[ids], training=False))
        values = hypothesis_scores(logits, kind)
        scores.extend(values.tolist())
        total_loss += float(np.sum(-values[np.arange(len(ids)), labels[ids]]/lengths[ids]))
        for reference, length, predicted in zip(packed[ids], lengths[ids], logits.argmax(-1), strict=True):
            sequence = collapse(predicted)
            decoded.append(sequence)
            edits += edit_distance(reference[:length].tolist(), sequence)
            target_count += int(length)
    measured = summary(labels[test_ids], np.asarray(scores).argmax(-1))
    measured.update(mean_ctc_nll_per_token=total_loss/len(test_ids),
        greedy_exact_sequences=sum(a == packed[i, :lengths[i]].tolist() for a, i in zip(decoded, test_ids, strict=True)),
        target_tokens=target_count, greedy_edits=edits, greedy_token_error_rate=edits/target_count)
    if weights_sha(classifier.weights) != classifier_hash:
        raise ValueError('Disconnected binary classifier changed')
    if weights_sha(encoder.weights) == encoder_before:
        raise ValueError('Encoder did not train')
    checkpoint = output/f'{fold}-{kind}.weights.h5'
    full.save_weights(checkpoint)
    result = {'summary': measured, 'parameters': full.count_params(), 'history': history,
        'draw_schedule_sha256': schedule.hexdigest(), 'test_ids': test_ids.tolist(),
        'train_ids': train_ids.tolist(), 'scores': scores, 'decoded': decoded,
        'encoder_weights_before_sha256': encoder_before,
        'encoder_weights_after_sha256': weights_sha(encoder.weights),
        'classifier_weights_sha256': classifier_hash, 'classifier_unchanged': True,
        'head_weights_sha256': weights_sha(head.weights), 'seconds': time.monotonic()-started,
        'checkpoint_file': checkpoint.name, 'checkpoint_sha256': sha(checkpoint)}
    (output/f'{fold}-{kind}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'fold': fold, 'kind': kind, 'complete': True, **measured}), flush=True)
    tf.keras.backend.clear_session()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    if args.output.exists():
        parser.error('Use a fresh result directory')
    if (any(sha(Path(__file__).parent/name) != digest for name, digest in plan['source_sha256'].items())
            or sha(args.initial_weights) != plan['initializer_sha256']
            or sha(args.features/'features.bin') != plan['feature_data_sha256']):
        raise ValueError('Source, initializer or scalar feature bytes differ')
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    sys.path.insert(0, str(args.upstream.resolve()))
    raw, labels, rows = load_bank(args.bank, args.features, plan['compact_bank'])
    raw, labels, rows = unique_bank(raw, labels, rows)
    if len(rows) != plan['unique_crops']:
        raise ValueError('Unique training crop count differs')
    x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    (args.output/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    ids = np.arange(len(rows), dtype=np.int32)
    folds, aggregate_predictions = {}, {kind: np.full(len(rows), -1, np.int32) for kind in ('words', 'phones')}
    engines = np.asarray([r['engine'] for r in rows])
    for fold in ('full-fit', *plan['compact_bank']['engines']):
        train_ids = ids if fold == 'full-fit' else ids[engines != fold]
        test_ids = ids if fold == 'full-fit' else ids[engines == fold]
        if fold != 'full-fit' and ({rows[i]['sha256'] for i in train_ids} & {rows[i]['sha256'] for i in test_ids}):
            raise ValueError('Held-engine fold has overlapping source waveform hashes')
        folds[fold] = {}
        for kind in ('words', 'phones'):
            result = fit(kind, fold, x, labels, rows, train_ids, test_ids,
                         plan, args.initial_weights, args.output)
            folds[fold][kind] = {'file': f'{fold}-{kind}.json', 'sha256': sha(args.output/f'{fold}-{kind}.json'),
                'summary': result['summary'], 'parameters': result['parameters'],
                'draw_schedule_sha256': result['draw_schedule_sha256'],
                'encoder_weights_before_sha256': result['encoder_weights_before_sha256'],
                'classifier_weights_sha256': result['classifier_weights_sha256']}
            if fold != 'full-fit':
                aggregate_predictions[kind][test_ids] = np.asarray(result['scores']).argmax(-1)
        left, right = folds[fold]['words'], folds[fold]['phones']
        if any(left[k] != right[k] for k in ('draw_schedule_sha256', 'encoder_weights_before_sha256', 'classifier_weights_sha256')):
            raise ValueError('Paired fold draws or initialization differ')
    aggregate = {kind: summary(labels, predicted) for kind, predicted in aggregate_predictions.items()}
    engine_macro = {kind: float(np.mean([folds[e][kind]['summary']['macro_accuracy']
                    for e in plan['compact_bank']['engines']])) for kind in aggregate}
    phone, word = aggregate['phones'], aggregate['words']
    gate = plan['diagnostic_gate']
    advance = (phone['macro_accuracy'] >= gate['minimum_macro_accuracy']
        and phone['nyx_recall'] >= gate['minimum_nyx_recall']
        and phone['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
        and engine_macro['phones'] >= engine_macro['words']+gate['minimum_engine_macro_gain']
        and phone['nyx_recall'] >= word['nyx_recall']-gate['maximum_nyx_recall_regression'])
    report = {'plan_sha256': sha(args.plan), 'rows_sha256': sha(args.output/'rows.json'),
        'unique_crops': len(rows), 'classes': CLASSES, 'word_tokens': WORD_TOKENS,
        'phone_tokens': TOKENS, 'phone_intent_strings': PHONE_STRINGS, 'folds': folds,
        'aggregate_held_engine': aggregate, 'engine_macro_accuracy': engine_macro,
        'advance': bool(advance), 'accepted': False, 'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'complete': True, 'advance': bool(advance), 'engine_macro_accuracy': engine_macro,
                      'aggregate': aggregate}), flush=True)


if __name__ == '__main__':
    main()
