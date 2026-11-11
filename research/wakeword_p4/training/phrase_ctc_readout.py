"""Training-only compact phrase readout feasibility on a frozen encoder.

No teacher phones, wake-validation scores, audio devices, export or firmware
are used. Held-engine rows are only held out from this new readout: the
encoder already saw the training corpus, so this is not independent testing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

CLASSES = ('nyx', 'mix', 'next', 'nick')
TOKENS = ('blank', 'hey', *CLASSES)
ALIASES = {'hey nyx': 0, 'hey nix': 0, 'hey nicks': 0,
           'hey mix': 1, 'hey next': 2, 'hey nick': 3}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weights_sha(variables):
    result = hashlib.sha256()
    for variable in variables:
        result.update(variable.numpy().tobytes())
    return result.hexdigest()


def phrase_class(row):
    text = row['text'].lower().strip().rstrip('.!?')
    if text not in ALIASES or row['label'] != int(ALIASES.get(text) == 0):
        raise ValueError('Unknown or conflicting synthesis-intent phrase label')
    return ALIASES[text]


def targets(classes):
    values = np.asarray(classes)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values >= 4)):
        raise ValueError('Expected four integer phrase classes')
    return np.column_stack((np.ones(len(values), np.int32), values.astype(np.int32)+2))


def hypothesis_scores(logits):
    """Equal-prior, equal-length two-token CTC hypotheses; blank is zero."""
    import tensorflow as tf
    values = tf.convert_to_tensor(logits, dtype=tf.float32)
    if values.shape.rank != 3 or values.shape[-1] != len(TOKENS) or values.shape[1] < 2:
        raise ValueError('Expected [batch,time,6] compact phrase logits')
    if not bool(tf.reduce_all(tf.math.is_finite(values))):
        raise ValueError('Nonfinite logits')
    batch, steps = tf.shape(values)[0], tf.shape(values)[1]
    repeated = tf.repeat(values, len(CLASSES), axis=0)
    labels = tf.tile(tf.constant(targets(np.arange(len(CLASSES)))), [batch, 1])
    losses = tf.nn.ctc_loss(labels=labels, logits=repeated,
        label_length=tf.fill([batch*len(CLASSES)], 2),
        logit_length=tf.fill([batch*len(CLASSES)], steps), logits_time_major=False, blank_index=0)
    return np.asarray(tf.reshape(-losses, [batch, len(CLASSES)]))


def summary(labels, predicted):
    matrix = np.zeros((4, 4), np.int64)
    for actual, prediction in zip(labels, predicted, strict=True):
        matrix[actual, prediction] += 1
    counts = matrix.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError('Each reported cohort must cover every phrase')
    recall = matrix.diagonal()/counts
    return {'confusion': matrix.tolist(), 'rows': int(matrix.sum()),
            'macro_accuracy': float(recall.mean()), 'recall_by_phrase': dict(zip(CLASSES, recall.tolist())),
            'nyx_recall': float(recall[0]),
            'confuser_false_nyx': int(matrix[1:, 0].sum()),
            'confuser_rows': int(counts[1:].sum()),
            'confuser_false_nyx_fraction': float(matrix[1:, 0].sum()/counts[1:].sum())}


def load_bank(bank_path, feature_path, plan):
    index_path = feature_path/'features.json'
    if (sha(index_path) != plan['feature_index_sha256']
            or sha(bank_path/'bank.json') != plan['bank_sha256']
            or sha(bank_path/'bank.npz') != plan['bank_data_sha256']):
        raise ValueError('Bank or feature-index identity differs')
    index = json.loads(index_path.read_bytes())
    by_key = {(row['path'], row['variant']): row for row in index}
    if len(by_key) != len(index):
        raise ValueError('Ambiguous original source/variant')
    reserved = {row['sha256'] for row in index if row['split'] != 'training'}
    meta = json.loads((bank_path/'bank.json').read_bytes())
    with np.load(bank_path/'bank.npz', allow_pickle=False) as data:
        features = data['features']
    if features.dtype != np.uint16 or features.shape != (len(meta['rows']), 335, 40):
        raise ValueError('Unexpected fixed bank geometry')
    raw = np.memmap(feature_path/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    rows, labels = [], []
    for i, row in enumerate(meta['rows']):
        original = by_key[row['path'], row['variant']]
        end = row['end_frame']
        if (original['split'] != 'training' or original['sha256'] in reserved
                or original['sha256'] != row['source_sha256'] or original['label'] != row['label']
                or row['label'] != 1-i % 2 or row['pair'] != i//2
                or type(end) is not int or not 335 <= end <= original['frames']
                or sha_array(features[i]) != row['feature_sha256']
                or not np.array_equal(features[i], raw[original['offset']+end-335:original['offset']+end])):
            raise ValueError('Training-only source, binary role or exact cached crop differs')
        path = Path(original['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe source identity')
        labels.append(phrase_class(original))
        rows.append({key: original[key] for key in ('path', 'variant', 'engine', 'sha256', 'text', 'label')})
    if len(rows) != plan['rows'] or sorted({row['engine'] for row in rows}) != plan['engines']:
        raise ValueError('Bank exposure differs from frozen plan')
    return features, np.asarray(labels, np.int32), rows


def sha_array(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def fit(cache, labels, train_ids, test_ids, plan, kind):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(plan['head_seed'])
    inputs = tf.keras.Input(cache.shape[1:])
    layer = tf.keras.layers.Dense(len(TOKENS) if kind == 'ctc' else len(CLASSES),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=plan['head_seed']))
    output = layer(inputs if kind == 'ctc' else tf.keras.layers.GlobalAveragePooling1D()(inputs))
    model = tf.keras.Model(inputs, output)
    optimizer = tf.keras.optimizers.Adam(plan['learning_rate'], global_clipnorm=5)
    packed = tf.constant(targets(labels))

    @tf.function
    def update(ids):
        with tf.GradientTape() as tape:
            logits = model(tf.gather(cache, ids), training=True)
            if kind == 'ctc':
                loss = tf.reduce_mean(tf.nn.ctc_loss(labels=tf.gather(packed, ids), logits=logits,
                    label_length=tf.fill([tf.shape(ids)[0]], 2),
                    logit_length=tf.fill([tf.shape(ids)[0]], cache.shape[1]),
                    logits_time_major=False, blank_index=0))/2
            else:
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                    tf.gather(labels, ids), logits, from_logits=True))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
        return loss

    rng, draws = np.random.default_rng(plan['draw_seed']), hashlib.sha256()
    history = []
    for step in range(1, plan['steps']+1):
        ids = rng.choice(train_ids, size=plan['batch_size'], replace=True).astype(np.int32)
        draws.update(ids.astype('<i4', copy=False).tobytes())
        loss = float(update(tf.constant(ids)))
        if not np.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        if step % 250 == 0 or step == plan['steps']:
            history.append({'step': step, 'loss': loss})
    values = np.asarray(model(cache[test_ids], training=False))
    scores = hypothesis_scores(values) if kind == 'ctc' else values
    predictions = scores.argmax(-1)
    measured = summary(labels[test_ids], predictions)
    if kind == 'ctc':
        from phone_sequences import collapse
        expected = targets(labels[test_ids]).tolist()
        decoded = [collapse(row) for row in values.argmax(-1)]
        measured.update(greedy_exact_sequence_fraction=float(np.mean([
            a == b for a, b in zip(decoded, expected, strict=True)])),
            mean_normalized_ctc_loss=float(-scores[np.arange(len(test_ids)), labels[test_ids]].mean()/2))
    return model, {'kind': kind, 'parameters': model.count_params(), 'draw_schedule_sha256': draws.hexdigest(),
                   'head_weights_sha256': weights_sha(model.weights), 'history': history,
                   'summary': measured,
                   'scores': scores.tolist(), 'row_ids': test_ids.tolist(), 'predicted': predictions.tolist()}


def passes(measured, gate):
    return (measured['macro_accuracy'] >= gate['minimum_macro_accuracy']
            and measured['nyx_recall'] >= gate['minimum_nyx_recall']
            and measured['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
            and ('minimum_greedy_exact_sequence_fraction' not in gate
                 or measured['greedy_exact_sequence_fraction'] >= gate['minimum_greedy_exact_sequence_fraction'])
            and ('maximum_mean_normalized_ctc_loss' not in gate
                 or measured['mean_normalized_ctc_loss'] <= gate['maximum_mean_normalized_ctc_loss']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'upstream', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh result directory')
    plan = json.loads(args.plan.read_bytes())
    source = Path(__file__).parent
    if any(sha(source/name) != digest for name, digest in plan['source_sha256'].items()):
        raise ValueError('Executed source differs from frozen plan')
    if sha(args.initial_weights) != plan['initializer_sha256']:
        raise ValueError('Initializer differs')
    import subprocess
    if (subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip()
            != plan['upstream_commit'] or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Upstream source differs')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from transfer import build_model
    features, labels, rows = load_bank(args.bank, args.features, plan)
    model, _ = build_model(335)
    model.load_weights(args.initial_weights)
    before = weights_sha(model.weights)
    temporal = tf.keras.layers.Reshape((67, 60))(model.get_layer('classifier').input)
    encoder = tf.keras.Model(model.input, temporal)
    cache = np.concatenate([np.asarray(encoder(
        np.clip(features[start:start+32].astype(np.float32)/25.6, 0, 26), training=False))
        for start in range(0, len(features), 32)])
    args.output.mkdir(parents=True, exist_ok=False)
    args.output.joinpath('plan.json').write_bytes(args.plan.read_bytes())
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'classes': CLASSES, 'tokens': TOKENS,
              'encoder_cache_sha256': sha_array(cache), 'binary_weights_before_sha256': before,
              'rows': rows, 'full_fit': {}, 'folds': {}, 'advance': False}
    ids = np.arange(len(labels), dtype=np.int32)
    for kind in ('mean', 'ctc'):
        head, result = fit(cache, labels, ids, ids, plan, kind)
        report['full_fit'][kind] = result
        head.save_weights(args.output/f'full-{kind}.weights.h5')
        print(json.dumps({'stage': 'full_fit', 'kind': kind, **result['summary']}), flush=True)
    if (report['full_fit']['mean']['draw_schedule_sha256']
            != report['full_fit']['ctc']['draw_schedule_sha256']):
        raise ValueError('Full-fit readout training draws differ')
    if passes(report['full_fit']['ctc']['summary'], plan['fit_gate']):
        joined = {kind: np.full(len(rows), -1, np.int32) for kind in ('mean', 'ctc')}
        for engine in plan['engines']:
            test = np.asarray([i for i, row in enumerate(rows) if row['engine'] == engine], np.int32)
            train = np.asarray([i for i, row in enumerate(rows) if row['engine'] != engine], np.int32)
            if {rows[i]['sha256'] for i in train} & {rows[i]['sha256'] for i in test}:
                raise ValueError('Held-engine fold shares source WAV with its readout training')
            fold = {}
            for kind in ('mean', 'ctc'):
                head, result = fit(cache, labels, train, test, plan, kind)
                head.save_weights(args.output/f'{engine}-{kind}.weights.h5')
                fold[kind] = result
                joined[kind][test] = result['predicted']
                print(json.dumps({'stage': 'held_engine', 'engine': engine, 'kind': kind,
                                  **result['summary']}), flush=True)
            if fold['mean']['draw_schedule_sha256'] != fold['ctc']['draw_schedule_sha256']:
                raise ValueError('Readout training draws differ')
            report['folds'][engine] = fold
        report['held_engine_aggregate'] = {kind: summary(labels, predicted) for kind, predicted in joined.items()}
        ctc, baseline = (report['held_engine_aggregate'][kind] for kind in ('ctc', 'mean'))
        report['advance'] = bool(passes(ctc, plan['readout_gate'])
            and ctc['macro_accuracy'] >= baseline['macro_accuracy']+plan['minimum_macro_gain_over_mean'])
    else:
        report['stopped_before_folds'] = 'Compact CTC readout fails the frozen full-bank fit gate.'
    after = weights_sha(model.weights)
    if before != after:
        raise ValueError('Readout fitting changed the frozen binary encoder/classifier')
    report['binary_weights_after_sha256'] = after
    report['binary_weights_unchanged'] = True
    report['limits'] = plan['limits']
    with (args.output/'result.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'advance': report['advance'], 'binary_weights_unchanged': True,
                      'held_engine_aggregate': report.get('held_engine_aggregate'),
                      'stopped_before_folds': report.get('stopped_before_folds')}), flush=True)


if __name__ == '__main__':
    main()
