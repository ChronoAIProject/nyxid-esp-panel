"""Training-only leave-engine-out ridge probes of fixed lexical embeddings.

These are six-class diagnostic decoders, never streaming wake detectors.
No existing trained detector is loaded and no validation/final data are scored.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

CLASSES = ['target', 'hey nick', 'hey next', 'hey nice', 'hey mix', 'hey nexus']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_weights(labels, engines):
    counts = Counter(zip(labels.tolist(), engines.tolist()))
    if any((label, engine) not in counts for label in range(6) for engine in set(engines)):
        raise ValueError('Every fitted engine must contain all six classes')
    return np.asarray([1. / (len(counts) * counts[(label, engine)])
                       for label, engine in zip(labels, engines)], dtype=np.float64)


def fit_ridge(x, labels, engines, penalty):
    """All standardization, intercepts and regression moments use fit rows only."""
    weights = balanced_weights(labels, engines)
    mean = weights @ x
    variance = weights @ ((x - mean) ** 2)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    z = (x - mean) / scale
    y = np.eye(6)[labels]
    intercept = weights @ y
    gram = z.T @ (z * weights[:, None])
    gram.flat[::len(gram) + 1] += penalty
    coefficients = solve(gram, z.T @ ((y - intercept) * weights[:, None]),
                         assume_a='pos', check_finite=True)
    return mean, scale, coefficients, intercept


def predict(x, fitted):
    mean, scale, coefficients, intercept = fitted
    return np.argmax(((x - mean) / scale) @ coefficients + intercept, axis=1)


def metrics(labels, predictions):
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    denominators = confusion.sum(axis=1)
    if np.any(denominators == 0):
        raise ValueError('Every diagnostic cohort must contain all six classes')
    recall = confusion.diagonal() / denominators
    return {'sources': len(labels), 'confusion_true_rows_predicted_columns': confusion.tolist(),
            'per_class_top1_fraction': recall.tolist(), 'macro_top1_fraction': float(recall.mean()),
            'pooled_top1_fraction': float(np.mean(labels == predictions)),
            'target_assigned_to_confuser_sources': int(confusion[0, 1:].sum()),
            'confuser_assigned_to_target_sources': int(confusion[1:, 0].sum())}


def load_inputs(data_root, manifest_path, plan):
    if sha(manifest_path) != plan['auxiliary_manifest_sha256']:
        raise ValueError('Frozen auxiliary manifest differs')
    feature_root = data_root / 'head-features-tts-v1'
    if sha(feature_root / 'index.jsonl') != plan['feature_index_sha256']:
        raise ValueError('Frozen original feature index differs')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['classes'] != CLASSES or not manifest['all_sources_training_only']
            or not manifest['heldout_source_hash_disjoint'] or manifest['audio_speed_changed']):
        raise ValueError('Auxiliary roles differ')
    rows = [json.loads(line) for line in (feature_root / 'index.jsonl').read_text().splitlines()]
    by_id = {row['id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Repeated original feature identity')
    heldout = {row['source_sha256'] for row in rows if row['split'] != 'training'}
    candidates = defaultdict(list)
    for record in manifest['records']:
        if record['variant'] != 0:
            continue
        row = by_id[record['id']]
        if (row['split'] != 'training' or row['kind'] != 'tts'
                or row['source_sha256'] in heldout
                or any(row[key] != record[key] for key in
                       ('source_sha256', 'features_sha256', 'engine', 'label', 'variant'))
                or type(record['class']) is not int or not 0 <= record['class'] < 6
                or bool(row['label']) != (record['class'] == 0)
                or not 25 <= record['minimum_end'] <= record['maximum_end'] <= row['frames']):
            raise ValueError('Original training role or warmed bounds differ')
        candidates[row['source_sha256']].append((record, row))
    chosen = []
    for source, aliases in sorted(candidates.items()):
        if len({(record['class'], record['engine']) for record, _ in aliases}) != 1:
            raise ValueError('A source has inconsistent lexical or engine roles')
        record, row = min(aliases, key=lambda pair: pair[0]['id'])
        path = Path(row['path'])
        path = path if path.is_absolute() else data_root / path
        if sha(path) != row['features_sha256']:
            raise ValueError('Original feature payload differs')
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.int8 or array.shape != (row['frames'], 96):
            raise ValueError('Original embedding format differs')
        end = (record['minimum_end'] + record['maximum_end']) // 2
        chosen.append((record | {'selected_end': end}, array[end - 16:end].astype(np.float64) / 128))
    if not chosen or len(chosen) != manifest['unique_source_recordings']:
        raise ValueError('Expected one original quiet view per unique auxiliary source')
    records, arrays = zip(*chosen)
    return list(records), np.stack(arrays)


def run(args):
    plan = json.loads(args.plan.read_text())
    if sha(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Frozen diagnostic source differs')
    if (plan['ridge_penalty'] != 1. or plan['seed'] != 741
            or plan['representations'] != ['full_history', 'time_mean']):
        raise ValueError('Expected fixed probe settings')
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    records, x = load_inputs(args.data_root, args.manifest, plan)
    labels = np.asarray([row['class'] for row in records])
    engines = np.asarray([row['engine'] for row in records])
    sources = np.asarray([row['source_sha256'] for row in records])
    if len(np.unique(sources)) != len(sources) or len(set(engines)) != 6:
        raise ValueError('Expected unique sources across six synthesis engines')
    rng = np.random.default_rng(plan['seed'])
    permuted = labels.copy()
    for engine in sorted(set(engines)):
        selected = np.flatnonzero(engines == engine)
        permuted[selected] = rng.permutation(labels[selected])
    args.output.mkdir(parents=True)
    results, predictions = [], []
    for name in plan['representations']:
        values = x.reshape(len(x), -1) if name == 'full_history' else x.mean(axis=1)
        for control, y in [('original_labels', labels), ('permuted_labels', permuted)]:
            folds = []
            for engine in sorted(set(engines)):
                test = engines == engine
                fit = ~test
                if set(sources[test]).intersection(sources[fit]):
                    raise ValueError('Source overlap across an excluded-engine fold')
                model = fit_ridge(values[fit], y[fit], engines[fit], plan['ridge_penalty'])
                fit_prediction = predict(values[fit], model)
                test_prediction = predict(values[test], model)
                fold = {'engine': str(engine), 'fit': metrics(y[fit], fit_prediction),
                        'excluded_engine': metrics(y[test], test_prediction)}
                folds.append(fold)
                predictions.append({'representation': name, 'labels': control, 'engine': str(engine),
                                    'sources': sources[test].tolist(), 'labels_expected': y[test].tolist(),
                                    'predictions': test_prediction.tolist()})
                print(json.dumps({'representation': name, 'labels': control, 'engine': str(engine),
                                  'macro_top1_fraction': fold['excluded_engine']['macro_top1_fraction']}), flush=True)
            results.append({'representation': name, 'labels': control, 'dimensions': values.shape[1],
                            'mean_excluded_engine_macro_top1_fraction': float(np.mean(
                                [fold['excluded_engine']['macro_top1_fraction'] for fold in folds])), 'folds': folds})
    private = args.output / 'rows-private.json'
    private.write_text(json.dumps({'records': records, 'predictions': predictions}, indent=2) + '\n')
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'script_sha256': sha(Path(__file__)),
              'auxiliary_manifest_sha256': sha(args.manifest), 'classes': CLASSES,
              'source_recordings': len(records), 'one_original_view_per_source': True,
              'all_sources_original_training_only': True, 'validation_and_final_scores_read': False,
              'source_hash_disjoint_folds': True, 'input_array_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
              'private_rows_sha256': sha(private), 'numpy_version': np.__version__,
              'speaker_playback': False, 'microphone_access': False, 'audio_speed_changed': False,
              'results': results, 'accepted': False,
              'limits': ['Closed six-class, already known word-window diagnosis; no streaming detector/FAR or recognition acceptance.',
                         'All sources are existing synthetic training recordings, not independent real speakers.',
                         'Original auxiliary energy/synthesis anchors are retained and are not phonetic ground truth.',
                         'Poor ridge transfer cannot prove an intrinsic representation limit or faulty source pronunciation.',
                         'Excluded-engine results cannot choose detector thresholds, operating windows or a deployment candidate.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
