"""Compare fixed-budget word-CTC sampling policies using training sources only."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from intent_phone_fit import fit, unique_bank
from intent_sampling import hierarchical_probabilities, distribution
from phrase_ctc_readout import load_bank, sha, summary

POLICIES = {'control': 'uniform-crop', 'balanced': 'engine-class-source-render-crop'}
REPLAY = ('summary', 'draw_schedule_sha256', 'encoder_weights_before_sha256',
          'encoder_weights_after_sha256', 'classifier_weights_sha256', 'head_weights_sha256',
          'scores', 'decoded', 'train_ids', 'test_ids', 'parameters')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def decision(aggregate, macros, gate):
    treatment, control = aggregate['balanced'], aggregate['control']
    return bool(treatment['macro_accuracy'] >= gate['minimum_macro_accuracy']
        and treatment['nyx_recall'] >= gate['minimum_nyx_recall']
        and treatment['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
        and macros['balanced'] >= macros['control']+gate['minimum_engine_macro_gain']
        and treatment['nyx_recall'] >= control['nyx_recall']-gate['maximum_nyx_recall_regression'])


def reference_probabilities(rows, ids):
    """Independent marginal counting with exact rational arithmetic."""
    selected = [rows[i] for i in ids]
    engines = {r['engine'] for r in selected}
    sources, renders, crops = {}, {}, Counter()
    for row in selected:
        group = row['engine'], row['class']
        sources.setdefault(group, set()).add(row['sha256'])
        source = (*group, row['sha256'])
        renders.setdefault(source, set()).add(row['variant'])
        crops[(*source, row['variant'])] += 1
    values = []
    for row in selected:
        group = row['engine'], row['class']
        source = (*group, row['sha256'])
        values.append(Fraction(1, len(engines)*4*len(sources[group])
            *len(renders[source])*crops[(*source, row['variant'])]))
    assert sum(values) == 1
    p = np.array([float(v) for v in values]); return p/p.sum()


def evaluate_result(root, plan, rows, labels, audit=False):
    engines = np.array([r['engine'] for r in rows]); all_ids = np.arange(len(rows), dtype=np.int32)
    predictions = {arm: np.full(len(rows), -1, np.int32) for arm in POLICIES}
    macros, records = {arm: [] for arm in POLICIES}, {}
    for fold in ('full-fit', *plan['compact_bank']['engines']):
        train_ids = all_ids if fold == 'full-fit' else all_ids[engines != fold]
        test_ids = all_ids if fold == 'full-fit' else all_ids[engines == fold]
        if fold != 'full-fit':
            assert not ({rows[i]['sha256'] for i in train_ids} & {rows[i]['sha256'] for i in test_ids})
        records[fold] = {}
        for arm in POLICIES:
            path = root/arm/f'{fold}-words.json'
            info = json.loads(path.read_bytes())
            assert info['train_ids'] == train_ids.tolist() and info['test_ids'] == test_ids.tolist()
            probabilities = None if arm == 'control' else reference_probabilities(rows, train_ids)
            saved = np.load(root/arm/f'{fold}-probabilities.npy', allow_pickle=False)
            expected = np.full(len(train_ids), 1/len(train_ids)) if probabilities is None else probabilities
            np.testing.assert_array_equal(saved, expected)
            digest, rng = hashlib.sha256(), np.random.default_rng(plan['draw_seed'])
            for _ in range(plan['steps']):
                draw = rng.choice(train_ids, plan['batch_size'], replace=True, p=probabilities).astype('<i4')
                digest.update(draw.tobytes())
            assert digest.hexdigest() == info['draw_schedule_sha256']
            scores = np.asarray(info['scores'])
            assert scores.shape == (len(test_ids), 4) and np.isfinite(scores).all()
            measured = summary(labels[test_ids], scores.argmax(-1))
            assert all(info['summary'][key] == value for key, value in measured.items())
            targets = [[1, int(labels[i])+2] for i in test_ids]
            assert len(info['decoded']) == len(targets)
            assert info['summary']['greedy_exact_sequences'] == sum(a == b for a, b in zip(targets, info['decoded'], strict=True))
            nll = float(np.mean(-scores[np.arange(len(test_ids)), labels[test_ids]]/2))
            assert abs(info['summary']['mean_ctc_nll_per_token']-nll) <= 1e-12
            assert info['classifier_unchanged'] and info['encoder_weights_before_sha256'] != info['encoder_weights_after_sha256']
            assert sha(root/arm/info['checkpoint_file']) == info['checkpoint_sha256']
            assert [r['step'] for r in info['history']] == [250, 500, 750, 1000]
            if arm == 'control':
                expected_control = plan['control_replay'][fold]
                for key in REPLAY:
                    assert canonical_sha(info[key]) == expected_control[key], ('Control replay differs', fold, key)
            if fold != 'full-fit':
                predictions[arm][test_ids] = scores.argmax(-1)
                macros[arm].append(measured['macro_accuracy'])
            records[fold][arm] = {'result_sha256': sha(path), 'checkpoint_sha256': info['checkpoint_sha256'],
                'summary': info['summary'], 'draw_schedule_sha256': digest.hexdigest()}
        left = json.loads((root/'control'/f'{fold}-words.json').read_bytes())
        right = json.loads((root/'balanced'/f'{fold}-words.json').read_bytes())
        for key in ('encoder_weights_before_sha256', 'classifier_weights_sha256', 'parameters'):
            assert left[key] == right[key], ('Initial model differs', fold, key)
    aggregate = {arm: summary(labels, pred) for arm, pred in predictions.items()}
    means = {arm: float(np.mean(values)) for arm, values in macros.items()}
    result = {'plan_sha256': sha(root.parent/'plan.json'), 'rows_sha256': sha(root/'rows.json'),
        'folds': records, 'aggregate_held_engine': aggregate, 'engine_macro_accuracy': means,
        'advance': decision(aggregate, means, plan['diagnostic_gate']), 'accepted': False,
        'limits': plan['limits']}
    if audit:
        assert result == json.loads((root/'result.json').read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('train', 'audit'), required=True)
    for name in ('artifacts', 'upstream', 'features', 'bank', 'initial-weights'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); root = args.artifacts.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    for name, digest in plan['source_sha256'].items():
        assert sha(Path(__file__).parent/name) == digest, ('Source differs', name)
    assert sha(args.initial_weights) == plan['initializer_sha256']
    assert sha(args.features/'features.bin') == plan['feature_data_sha256']
    assert subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() == plan['upstream_commit']
    assert not subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])
    raw, labels, rows = load_bank(args.bank, args.features, plan['compact_bank'])
    raw, labels, rows = unique_bank(raw, labels, rows)
    assert len(rows) == plan['unique_crops']
    study = root/'study'
    if args.mode == 'train':
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(3)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        sys.path.insert(0, str(args.upstream.resolve()))
        study.mkdir(exist_ok=False)
        (study/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
        for arm in POLICIES:
            (study/arm).mkdir()
        ids = np.arange(len(rows), dtype=np.int32); engines = np.array([r['engine'] for r in rows])
        x = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        exposure = {}
        for fold in ('full-fit', *plan['compact_bank']['engines']):
            train_ids = ids if fold == 'full-fit' else ids[engines != fold]
            test_ids = ids if fold == 'full-fit' else ids[engines == fold]
            exposure[fold] = {}
            for arm, policy in POLICIES.items():
                p = None if arm == 'control' else hierarchical_probabilities(rows, train_ids)
                expected = np.full(len(train_ids), 1/len(train_ids)) if p is None else p
                np.save(study/arm/f'{fold}-probabilities.npy', expected, allow_pickle=False)
                exposure[fold][arm] = distribution(rows, train_ids, p)
                fitted = fit('words', fold, x, labels, rows, train_ids, test_ids,
                    {**plan, 'intent_sampling': policy}, args.initial_weights, study/arm)
                if arm == 'control':
                    for key in REPLAY:
                        assert canonical_sha(fitted[key]) == plan['control_replay'][fold][key], ('Control replay differs', fold, key)
                print(json.dumps({'fold_complete': fold, 'arm': arm}), flush=True)
        (study/'exposure.json').write_text(json.dumps(exposure, indent=2)+'\n')
    else:
        assert rows == json.loads((study/'rows.json').read_bytes())
    result = evaluate_result(study, plan, rows, labels, audit=args.mode == 'audit')
    output = root/'audit.json' if args.mode == 'audit' else study/'result.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'phase': args.mode, 'advance': result['advance'],
        'engine_macro_accuracy': result['engine_macro_accuracy'], 'aggregate': result['aggregate_held_engine']}), flush=True)


if __name__ == '__main__':
    main()
