"""Reconstruct raw greedy phone observations and fixed-triplet comparisons."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import wave

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def distance(left, right):
    table = np.zeros((len(left)+1, len(right)+1), np.int64)
    table[:, 0] = np.arange(len(left)+1); table[0] = np.arange(len(right)+1)
    for i, a in enumerate(left, 1):
        for j, b in enumerate(right, 1):
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+(a != b))
    return int(table[-1, -1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'neighbours', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh audit output')
    root = args.artifacts.resolve(); read = lambda name: json.loads((root/name).read_bytes())
    plan, result, loading = read('plan.json'), read('result.json'), read('model-loading.json')
    assert result['plan_sha256'] == sha(root/'plan.json') and result['accepted'] is False
    assert all(not value for value in loading.values())
    for name, expected in plan['source_sha256'].items():
        assert sha(root/'source'/name) == expected
    for name, expected in plan['neighbour_inputs_sha256'].items():
        assert sha(args.neighbours/name) == expected
    for name, expected in plan['model_files_sha256'].items():
        if name.endswith('.json'):
            assert sha(root/'model-config'/name) == expected
    for name, expected in result['evidence_sha256'].items():
        assert sha(root/name) == expected
    assert sha(args.features/'features.json') == plan['feature_index_sha256']
    config = read('model-config/config.json'); vocabulary = read('model-config/vocab.json')
    names = {v: k for k, v in vocabulary.items()}
    assert config['pad_token_id'] == 0 and not config.get('add_adapter', False)
    review = json.loads((args.neighbours/'review-candidates.json').read_bytes())
    inventory = json.loads((args.neighbours/'inventory.json').read_bytes())
    ids = sorted({row[k] for row in review for k in ('anchor_id', 'positive_id', 'negative_id')})
    entries = [json.loads(line) for line in (root/'predictions.jsonl').read_text().splitlines()]
    assert [row['id'] for row in entries] == ids and len(entries) == plan['source_paths']
    rows = json.loads((args.features/'features.json').read_bytes())
    by_path = {(row['path'], row['variant']): row for row in rows}
    reserved = {row['sha256'] for row in rows if row['split'] != 'training'}
    logits = np.load(root/'logits.npz', allow_pickle=False)
    assert set(logits.files) == {f'source_{uid}' for uid in ids}
    predictions, frames, classes = {}, 0, defaultdict(list)
    for item in entries:
        uid = item['id']; source = inventory[uid]; original = by_path[source['path'], 0]
        for key in ('path', 'engine', 'class', 'text', 'source_sha256'):
            assert item[key] == source[key]
        assert original['split'] == 'training' and original['sha256'] not in reserved
        path = args.features/'dataset'/item['path']
        assert sha(path) == item['source_sha256'] == original['sha256']
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, item['samples'])
        count = item['samples']
        for kernel, stride in zip(config['conv_kernel'], config['conv_stride'], strict=True):
            count = (count-kernel)//stride+1
        value = logits[f'source_{uid}']
        assert value.dtype == np.float32 and value.shape == (count, len(vocabulary))
        assert list(value.shape) == item['logits_shape'] and np.isfinite(value).all()
        assert hashlib.sha256(value.tobytes()).hexdigest() == item['logits_sha256']
        decoded = [int(key) for key, _ in itertools.groupby(value.argmax(-1)) if key != 0]
        assert decoded == item['tokens'] and [names[i] for i in decoded] == item['phones']
        predictions[uid] = decoded; frames += count
        classes[tuple(decoded)].append(uid)
    compared, cohorts = [], defaultdict(Counter)
    for row in review:
        a, p, n = [predictions[row[k]] for k in ('anchor_id', 'positive_id', 'negative_id')]
        if any(not sequence for sequence in (a, p, n)):
            positive = negative = margin = None; order = 'empty_prediction'
        else:
            pd, nd = Fraction(distance(a, p), max(len(a), len(p))), Fraction(distance(a, n), max(len(a), len(n)))
            positive, negative = float(pd), float(nd); margin = negative-positive
            order = 'negative_closer' if nd < pd else 'positive_closer' if nd > pd else 'tie'
        compared.append({**row, 'phone_order': order, 'phone_positive_distance': positive,
            'phone_negative_distance': negative, 'phone_margin': margin})
        cohorts['spectral_inversion' if row['margin'] < 0 else 'spectral_noninversion'][order] += 1
    assert compared == read('comparisons.json')
    assert {k: dict(v) for k, v in cohorts.items()} == result['phone_order_by_spectral_cohort']
    conflicts = [group for group in classes.values() if len({inventory[uid]['class'] for uid in group}) > 1]
    assert conflicts == result['different_intent_identical_predicted_sequence_groups']
    assert sum(not tokens for tokens in predictions.values()) == result['empty_predictions']
    assert len(review) == result['review_triplets'] == plan['review_triplets']
    audit = {'result_sha256': sha(root/'result.json'), 'auditor_sha256': sha(Path(__file__)),
        'sources': len(entries), 'triplets': len(compared), 'frame_logits': frames*len(vocabulary),
        'all_original_wav_hashes_rates_and_lengths_verified': True,
        'all_logits_and_greedy_ids_phone_strings_verified': True,
        'all_edit_orders_cohorts_and_prediction_collisions_reconstructed': True,
        'accepted': False,
        'limits': 'Saved-output accounting, not independent model inference, human pronunciation truth or wake-quality acceptance.'}
    args.output.write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
