"""Unprompted phoneme consistency on a frozen training-only review queue.

Recognizer outputs are retained as fallible observations, never wake labels.
Original16k PCM is read without cropping, resampling, augmentation or playback.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

import numpy as np

from phone_sequences import collapse, edit_distance
from phrase_ctc_readout import phrase_class, sha


def selected_ids(inventory, review):
    ids = set()
    if [r['id'] for r in inventory] != list(range(len(inventory))):
        raise ValueError('Ambiguous review inventory IDs')
    for row in review:
        triple = [row[k] for k in ('anchor_id', 'positive_id', 'negative_id')]
        if any(type(i) is not int or not 0 <= i < len(inventory) for i in triple):
            raise ValueError('Invalid review source ID')
        if (type(row['class']) is not int or inventory[triple[0]]['class'] != 0 or inventory[triple[1]]['class'] != 0
                or inventory[triple[2]]['class'] != row['class'] or row['class'] not in (1, 2, 3)
                or len({inventory[i]['source_sha256'] for i in triple}) != 3
                or any(inventory[i]['voice_group'] != inventory[triple[0]]['voice_group'] for i in triple)):
            raise ValueError('Review triplet roles, hashes or voice groups differ')
        ids.update(triple)
    return sorted(ids)


def triplet_comparisons(review, predictions):
    result = []
    for row in review:
        a, p, n = [predictions[row[k]]['tokens'] for k in ('anchor_id', 'positive_id', 'negative_id')]
        if any(not value for value in (a, p, n)):
            result.append({**row, 'phone_order': 'empty_prediction', 'phone_positive_distance': None,
                'phone_negative_distance': None, 'phone_margin': None})
            continue
        positive = edit_distance(a, p)/max(len(a), len(p))
        negative = edit_distance(a, n)/max(len(a), len(n))
        # Exact rational order; avoid spurious float ties in reporting.
        gap = edit_distance(a, n)*max(len(a), len(p))-edit_distance(a, p)*max(len(a), len(n))
        order = 'negative_closer' if gap < 0 else 'positive_closer' if gap > 0 else 'tie'
        result.append({**row, 'phone_order': order, 'phone_positive_distance': positive,
            'phone_negative_distance': negative, 'phone_margin': negative-positive})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'neighbours', 'features', 'model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    plan = json.loads(args.plan.read_bytes())
    for name, expected in plan['source_sha256'].items():
        assert sha(Path(__file__).parent/name) == expected, ('Source differs', name)
    for name, expected in plan['neighbour_inputs_sha256'].items():
        assert sha(args.neighbours/name) == expected, ('Review input differs', name)
    assert sha(args.features/'features.json') == plan['feature_index_sha256']
    for name, expected in plan['model_files_sha256'].items():
        assert sha(args.model/name) == expected, ('Recognizer input differs', name)
    for name, expected in plan['runtime'].items():
        assert version(name) == expected, ('Runtime differs', name)
    inventory = json.loads((args.neighbours/'inventory.json').read_bytes())
    review = json.loads((args.neighbours/'review-candidates.json').read_bytes())
    assert len(review) == plan['review_triplets']
    ids = selected_ids(inventory, review)
    assert len(ids) == plan['source_paths']
    assert len({inventory[i]['source_sha256'] for i in ids}) == plan['source_hashes']
    index = json.loads((args.features/'features.json').read_bytes())
    lookup = {(r['path'], r['variant']): r for r in index}
    assert len(lookup) == len(index)
    reserved = {r['sha256'] for r in index if r['split'] != 'training'}
    for uid in ids:
        item = inventory[uid]; row = lookup[item['path'], 0]; path = Path(item['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert row['split'] == 'training' and row['sha256'] not in reserved
        assert row['sha256'] == item['source_sha256'] and phrase_class(row) == item['class']
    args.output.mkdir(parents=True, exist_ok=False); (args.output/'source').mkdir()
    (args.output/'model-config').mkdir()
    for name in plan['source_sha256']:
        shutil.copyfile(Path(__file__).parent/name, args.output/'source'/name)
    for name in plan['model_files_sha256']:
        if name.endswith('.json'):
            shutil.copyfile(args.model/name, args.output/'model-config'/name)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[3], text=True).strip()
    launch = {'checkout_revision': revision, 'command': [sys.executable, *sys.argv],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'runtime': {n: version(n) for n in plan['runtime']},
        'scope': 'CPU inference on original training-source WAV files, without text prompts or audio devices.'}
    (args.output/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model, loading = Wav2Vec2ForCTC.from_pretrained(args.model, local_files_only=True,
        weights_only=True, output_loading_info=True)
    (args.output/'model-loading.json').write_text(json.dumps(loading, indent=2)+'\n')
    assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
    model.eval()
    assert model.config.pad_token_id == 0 and str(next(model.parameters()).device) == 'cpu'
    vocabulary = json.loads((args.model/'vocab.json').read_bytes())
    assert set(vocabulary.values()) == set(range(model.config.vocab_size))
    names = {v: k for k, v in vocabulary.items()}
    predictions, logits = {}, {}
    with (args.output/'predictions.jsonl').open('x') as log, torch.inference_mode():
        for number, uid in enumerate(ids, 1):
            item = inventory[uid]; row = lookup[item['path'], 0]
            wav_path = args.features/'dataset'/item['path']
            assert sha(wav_path) == item['source_sha256'], ('Original WAV differs', item['path'])
            with wave.open(str(wav_path), 'rb') as wav:
                assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (16000, 1, 2, row['samples'])
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)/32768
            inputs = extractor(pcm, sampling_rate=16000, return_tensors='pt')
            values = model(**inputs).logits[0].cpu().numpy().astype(np.float32, copy=False)
            assert values.ndim == 2 and values.shape[1] == len(vocabulary) and np.isfinite(values).all()
            tokens = collapse(values.argmax(-1))
            result = {k: item[k] for k in ('id', 'path', 'engine', 'class', 'text', 'source_sha256')}
            result.update(samples=len(pcm), tokens=tokens, phones=[names[i] for i in tokens],
                logits_shape=list(values.shape), logits_sha256=hashlib.sha256(values.tobytes()).hexdigest())
            predictions[uid] = result; logits[f'source_{uid}'] = values
            log.write(json.dumps(result, ensure_ascii=False)+'\n'); log.flush()
            if number % 12 == 0 or number == len(ids):
                print(json.dumps({'completed': number, 'total': len(ids)}), flush=True)
    np.savez_compressed(args.output/'logits.npz', **logits)
    comparisons = triplet_comparisons(review, predictions)
    (args.output/'comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    categories = defaultdict(Counter)
    for row in comparisons:
        categories['spectral_inversion' if row['margin'] < 0 else 'spectral_noninversion'][row['phone_order']] += 1
    sequences = defaultdict(list)
    for value in predictions.values():
        sequences[tuple(value['tokens'])].append(value['id'])
    collisions = [uids for uids in sequences.values() if len({predictions[uid]['class'] for uid in uids}) > 1]
    report = {'scope': plan['scope'], 'plan_sha256': sha(args.plan),
        'source_commit': revision, 'source_paths': len(predictions), 'review_triplets': len(comparisons),
        'phone_order_by_spectral_cohort': {k: dict(v) for k, v in categories.items()},
        'different_intent_identical_predicted_sequence_groups': collisions,
        'empty_predictions': sum(not row['tokens'] for row in predictions.values()), 'accepted': False,
        'evidence_sha256': {n: sha(args.output/n) for n in ('launch.json', 'model-loading.json', 'predictions.jsonl', 'logits.npz', 'comparisons.json')},
        'limits': plan['limits']}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
