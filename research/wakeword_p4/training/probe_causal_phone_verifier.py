"""Check phone cues available at the first wake score crossing on development audio."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from stream_data import features, render_stream

THRESHOLD = .994
LOOKBACK = 38400  # 2.4 s; fixed relative to trigger, not the phrase annotation.
EXTRA = 3840      # Report zero lookahead and the cost of 240 ms separately.
MODEL = '921081f23ef620c840b2e397891602d52e1e3e5b8e33bfc8c0ccc8e9e18c0383'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def evidence(phones):
    # Fixed from training pronunciations, not fitted to development scores.
    return any(a == 'h' and b in ('eɪ', 'eː', 'e', 'aɪ') and
               (c, d, e, f) == ('n', 'ɪ', 'k', 's')
               for a, b, c, d, e, f in zip(*(phones[i:] for i in range(6))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'validation', 'backgrounds', 'phone-model', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    plan_path = Path(__file__).parent/'plans/microwake-near-word-errors-v1.json'
    plan = json.loads(plan_path.read_bytes())
    if (sha(args.backgrounds/'manifest.json') != plan['background_manifest_sha256'] or
            any(sha(args.phone_model/name) != val for name, val in plan['phoneme_model_files'].items()) or
            any(version(name) != val for name, val in plan['runtime_packages'].items())):
        raise ValueError('Use the original render backgrounds and pinned local phone model')
    calibration_path = args.validation/'threshold.json'
    details_path = args.validation/'validation-details.json'
    calibration = json.loads(calibration_path.read_bytes())
    index = args.features/'features.json'
    if (sha(index) != calibration['feature_index_sha256'] or calibration['model_sha256'] != MODEL
            or calibration['smoothing_steps'] != 5):
        raise ValueError('Use the frozen mined model validation')
    details = json.loads(details_path.read_bytes())
    raw_index = json.loads(index.read_bytes())
    others = {r['sha256'] for r in raw_index if r['split'] != 'validation'}
    positives, negatives = defaultdict(list), []
    for row, scored in zip((r for r in raw_index if r['split'] == 'validation'), details['validation'], strict=True):
        if (row['path'], row['variant'], row['label']) != (scored['source'], scored['variant'], scored['label']):
            raise ValueError('Score order mismatch')
        if scored['max_score'] < THRESHOLD:
            continue
        if row['sha256'] in others:
            raise ValueError('Reserved split source overlap')
        pos = np.flatnonzero(np.asarray(scored['scores']) >= THRESHOLD)
        if not len(pos):
            raise ValueError('Missing first crossing')
        # Include all false positives and a deterministic positive sample of
        # at most twelve source recordings per engine, with one view each.
        trigger = (3*(int(pos[0])+4)+5)*160
        if row['label']:
            positives[row['engine']].append((row, trigger))
        else:
            negatives.append((row, trigger))
    selected = list(negatives)
    for group in positives.values():
        seen = set()
        for row, at in sorted(group, key=lambda r: hashlib.sha256(
                f"{r[0]['sha256']}:{r[0]['variant']}".encode()).hexdigest()):
            if row['sha256'] not in seen:
                selected.append((row, at))
                seen.add(row['sha256'])
            if len(seen) == 12:
                break

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.phone_model, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(args.phone_model, local_files_only=True, weights_only=True).eval()
    tokens = {v: k for k, v in json.loads((args.phone_model/'vocab.json').read_bytes()).items()}

    @torch.inference_mode()
    def recognize(pcm):
        data = extractor(pcm.astype(np.float32)/32768, sampling_rate=16000, return_tensors='pt')
        ids = torch.unique_consecutive(model(**data).logits[0].argmax(dim=-1)).tolist()
        return [tokens[i] for i in ids if i != model.config.pad_token_id]

    backgrounds = [str(args.backgrounds/r['path']) for r in json.loads((args.backgrounds/'manifest.json').read_bytes())['records']
                   if r['split'] == 'validation']
    cache = np.memmap(args.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    dataset = args.features/'dataset'
    generation = json.loads((dataset/'generation.json').read_bytes())
    result, counts = [], defaultdict(Counter)
    for n, (row, at) in enumerate(selected, 1):
        path = Path(row['path'])
        if path.is_absolute() or '..' in path.parts or sha(dataset/path) != row['sha256']:
            raise ValueError('Expected intact validation source')
        source, rate = sf.read(dataset/path, dtype='int16')
        if rate != 16000 or source.ndim != 1:
            raise ValueError('Expected unchanged 16 kHz mono')
        seed = int(hashlib.sha256(row['path'].encode()).hexdigest()[:8], 16)+row['variant']
        wave, lead = render_stream(row, dataset, np.random.default_rng(seed), row['variant'], backgrounds, generation['lead_seconds'])
        if lead//160 != row['lead_frame'] or not np.array_equal(features(wave), cache[row['offset']:row['offset']+row['frames']]):
            raise ValueError('Render failed to reproduce original cached detector input')
        if not 0 < at <= len(wave):
            raise ValueError('Trigger clock outside rendered PCM')
        wave = np.rint(wave).astype(np.int16)
        lo = max(0, at-LOOKBACK)
        before = recognize(wave[lo:at])
        later = recognize(wave[lo:min(len(wave), at+EXTRA)])
        full = recognize(source)
        name = 'positive' if row['label'] else row['text'].lower().strip('.!?')
        for key, phones in (('full_source', full), ('at_trigger', before), ('plus_240ms', later)):
            counts[name].update({'views' if key == 'full_source' else 'views_'+key: 1,
                                 'target_'+key: evidence(phones),
                                 'm_'+key: 'm' in phones, 't_'+key: 't' in phones})
        result.append({'source_sha256': row['sha256'], 'engine': row['engine'], 'text': row['text'],
                       'variant': row['variant'], 'label': row['label'], 'trigger_sample': at,
                       'full_source': full, 'at_trigger': before, 'plus_240ms': later})
        if n % 20 == 0:
            print(f'{n}/{len(selected)}', flush=True)
    report = {'scope': 'Trigger-conditioned diagnostic on existing synthetic development audio', 'wake_model_sha256': MODEL,
              'validation_details_sha256': sha(details_path), 'threshold': THRESHOLD,
              'lookback_samples': LOOKBACK, 'extra_samples': EXTRA, 'counts': counts, 'rows': result,
              'limits': 'One large offline phoneme model, fixed exact-sequence readout; neither deployable verifier nor human pronunciation/latency evidence.'}
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
