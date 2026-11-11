"""Read-only reconstruction of the shared-phone versus word-token diagnostic."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_compact_study import require, sha


def classification(labels, predicted):
    confusion = np.zeros((4, 4), np.int64)
    np.add.at(confusion, (labels, predicted), 1)
    support = confusion.sum(axis=1)
    require(np.all(support > 0), 'Missing intent class')
    recalls = confusion.diagonal()/support
    return {'confusion': confusion.tolist(), 'rows': int(confusion.sum()),
        'macro_accuracy': float(recalls.mean()),
        'recall_by_phrase': dict(zip(('nyx', 'mix', 'next', 'nick'), recalls.tolist())),
        'nyx_recall': float(recalls[0]), 'confuser_false_nyx': int(confusion[1:, 0].sum()),
        'confuser_rows': int(support[1:].sum()),
        'confuser_false_nyx_fraction': float(confusion[1:, 0].sum()/support[1:].sum())}


def distance(a, b):
    table = np.zeros((len(a)+1, len(b)+1), np.int64)
    table[:, 0] = np.arange(len(a)+1)
    table[0] = np.arange(len(b)+1)
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+(a[i-1] != b[j-1]))
    return int(table[-1, -1])


def audit(args):
    root, repo, features = args.artifacts.resolve(), args.repository.resolve(), args.features.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'launch.json').read_bytes())
    committed = subprocess.check_output(['git', 'show', launch['checkout_revision']+
        ':research/wakeword_p4/training/plans/intent-phone-generalization-v1.json'], cwd=repo)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256'],
            'Committed plan/launch differ')
    require(sha(root/'study/plan.json') == sha(root/'plan.json'), 'Executed plan differs')
    require(sha(args.initial_weights) == plan['initializer_sha256'], 'Initializer differs')
    require(sha(features/'features.json') == plan['compact_bank']['feature_index_sha256']
            and sha(features/'features.bin') == plan['feature_data_sha256'], 'Feature bytes differ')
    for name, digest in plan['source_sha256'].items():
        require(sha(root/'source'/name) == launch['source_sha256'][name] == digest, 'Source differs: '+name)
    for name, digest in plan['audit_source_sha256'].items():
        require(sha(root/'auditing'/name) == digest, 'Audit source differs: '+name)
    command = launch['command']
    require(Path(command[1]).resolve() == root/'source/intent_phone_fit.py'
            and command[2::2] == ['--plan', '--upstream', '--features', '--bank', '--initial-weights', '--output'],
            'Training command differs')
    for name, expected in (('--plan', root/'plan.json'), ('--features', features), ('--bank', args.bank),
                           ('--initial-weights', args.initial_weights), ('--output', root/'study')):
        require(Path(command[command.index(name)+1]).resolve() == expected.resolve(), 'Launch input differs: '+name)
    require(subprocess.check_output(['git', '-C', command[command.index('--upstream')+1], 'rev-parse', 'HEAD'],
            text=True).strip() == plan['upstream_commit'], 'Upstream revision differs')
    sys.path.insert(0, str(root/'source'))
    from phrase_ctc_readout import load_bank
    raw, original_labels, original_rows = load_bank(args.bank, features, plan['compact_bank'])
    actual = json.loads((root/'study/rows.json').read_bytes())
    # Rebuild unique groups independently; check each retained crop against
    # the original fully verified bank, not just the new trainer metadata.
    groups, expected = {}, []
    for index, (crop, label, source) in enumerate(zip(raw, original_labels, original_rows, strict=True)):
        digest = hashlib.sha256(crop.astype('<u2', copy=False).tobytes()).hexdigest()
        if digest not in groups:
            groups[digest] = len(expected)
            expected.append({**source, 'class': int(label), 'crop_sha256': digest, 'bank_rows': []})
        item = expected[groups[digest]]
        require(all(item[k] == source[k] for k in ('engine', 'sha256', 'variant')) and item['class'] == label,
                'Conflicting source/intent in repeated crop')
        item['bank_rows'].append(index)
    require(actual == expected and len(actual) == plan['unique_crops'], 'Unique crop inventory differs')
    result = json.loads((root/'study/result.json').read_bytes())
    require(result['plan_sha256'] == sha(root/'plan.json')
            and result['rows_sha256'] == sha(root/'study/rows.json') and result['accepted'] is False,
            'Result binding differs')
    labels = np.array([r['class'] for r in actual], np.int32)
    engines = np.array([r['engine'] for r in actual])
    ids = np.arange(len(actual), dtype=np.int32)
    references = {'words': [[1, 2], [1, 3], [1, 4], [1, 5]],
                  'phones': [[1, 2, 3, 4, 5, 6], [1, 2, 7, 4, 5, 6],
                             [1, 2, 3, 8, 5, 6, 9], [1, 2, 3, 4, 5]]}
    aggregate = {kind: np.full(len(ids), -1, np.int32) for kind in references}
    engine_macro = {kind: [] for kind in references}
    schedules, reports, initials = {}, {}, set()
    for fold in ('full-fit', *plan['compact_bank']['engines']):
        train_ids = ids if fold == 'full-fit' else ids[engines != fold]
        test_ids = ids if fold == 'full-fit' else ids[engines == fold]
        if fold != 'full-fit':
            require(not ({actual[i]['sha256'] for i in train_ids} & {actual[i]['sha256'] for i in test_ids}),
                    'Held-engine source overlap')
        digest, rng = hashlib.sha256(), np.random.default_rng(plan['draw_seed'])
        for _ in range(plan['steps']):
            draw = rng.choice(train_ids, size=plan['batch_size'], replace=True).astype('<i4')
            digest.update(draw.tobytes())
        schedules[fold] = digest.hexdigest()
        for kind in references:
            info = result['folds'][fold][kind]
            path = root/'study'/f'{fold}-{kind}.json'
            require(info['file'] == path.name and info['sha256'] == sha(path), 'Fold result differs')
            measured = json.loads(path.read_bytes())
            require(measured['train_ids'] == train_ids.tolist() and measured['test_ids'] == test_ids.tolist()
                    and measured['draw_schedule_sha256'] == info['draw_schedule_sha256'] == digest.hexdigest(),
                    'Exact fold draws/coverage differ')
            require([h['step'] for h in measured['history']]
                    == list(range(250, plan['steps']+1, 250)), 'Unplanned step/checkpoint selection')
            require(measured['classifier_unchanged'] and measured['encoder_weights_before_sha256']
                    != measured['encoder_weights_after_sha256'], 'Wrong trainable scope')
            initials.add((measured['encoder_weights_before_sha256'], measured['classifier_weights_sha256']))
            require(sha(root/'study'/measured['checkpoint_file']) == measured['checkpoint_sha256'],
                    'Checkpoint bytes differ')
            scores = np.asarray(measured['scores'])
            require(scores.shape == (len(test_ids), 4) and np.isfinite(scores).all(), 'Invalid fold scores')
            predicted = scores.argmax(-1)
            reconstruction = classification(labels[test_ids], predicted)
            correct, edits, tokens = 0, 0, 0
            lengths = []
            for index, decoded in zip(test_ids, measured['decoded'], strict=True):
                reference = references[kind][labels[index]]
                require(all(type(v) is int and 0 < v < (6 if kind == 'words' else 10) for v in decoded),
                        'Invalid greedy CTC token sequence')
                correct += decoded == reference
                edits += distance(reference, decoded)
                tokens += len(reference)
                lengths.append(len(reference))
            reconstruction.update(mean_ctc_nll_per_token=float(np.mean(-scores[np.arange(len(test_ids)), labels[test_ids]]/lengths)),
                greedy_exact_sequences=correct, target_tokens=tokens, greedy_edits=edits,
                greedy_token_error_rate=edits/tokens)
            for key, value in reconstruction.items():
                observed = measured['summary'][key]
                require(abs(value-observed) <= 1e-12 if isinstance(value, float) else value == observed,
                        'Reconstructed fold metric differs: '+fold+' '+kind+' '+key)
            require(info['summary'] == measured['summary'], 'Fold summary reference differs')
            if fold != 'full-fit':
                aggregate[kind][test_ids] = predicted
                engine_macro[kind].append(reconstruction['macro_accuracy'])
            reports[fold+':'+kind] = {'summary': reconstruction,
                'result_sha256': sha(path), 'checkpoint_sha256': measured['checkpoint_sha256']}
    require(len(initials) == 1, 'Fold initial binary weights differ')
    reconstructed_aggregate = {kind: classification(labels, predicted) for kind, predicted in aggregate.items()}
    macros = {kind: float(np.mean(values)) for kind, values in engine_macro.items()}
    require(reconstructed_aggregate == result['aggregate_held_engine'] and macros == result['engine_macro_accuracy'],
            'Aggregate/macro accounting differs')
    phone, word, gate = reconstructed_aggregate['phones'], reconstructed_aggregate['words'], plan['diagnostic_gate']
    advance = (phone['macro_accuracy'] >= gate['minimum_macro_accuracy']
        and phone['nyx_recall'] >= gate['minimum_nyx_recall']
        and phone['confuser_false_nyx_fraction'] <= gate['maximum_confuser_false_nyx_fraction']
        and macros['phones'] >= macros['words']+gate['minimum_engine_macro_gain']
        and phone['nyx_recall'] >= word['nyx_recall']-gate['maximum_nyx_recall_regression'])
    require(bool(advance) == result['advance'], 'Diagnostic advancement decision differs')
    return {'plan_sha256': sha(root/'plan.json'), 'source_commit': launch['checkout_revision'],
        'source_sha256': sha(Path(__file__)), 'result_sha256': sha(root/'study/result.json'),
        'unique_crops': len(actual), 'distinct_source_hashes': len({r['sha256'] for r in actual}),
        'unique_crops_by_engine_class': dict(sorted(Counter(r['engine']+':'+str(r['class']) for r in actual).items())),
        'all_source_crops_folds_schedules_metrics_and_gate_reconstructed': True,
        'original_binary_initialization_identical_and_classifiers_unchanged': True,
        'draw_schedule_sha256': schedules, 'folds': reports,
        'aggregate_held_engine': reconstructed_aggregate, 'engine_macro_accuracy': macros,
        'advance': bool(advance), 'accepted': False,
        'limits': 'Training-only source/score reconstruction. It does not independently replay all optimizer updates. Initializer already saw the original training corpus, so held-engine folds exclude only this diagnostic fine-tune, not original encoder exposure. No independent pronunciation, open-set wake FAR, latency, MCU resource or call-stability acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'features', 'bank', 'initial-weights', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('folds', 'unique_crops_by_engine_class')}, indent=2))


if __name__ == '__main__':
    main()
