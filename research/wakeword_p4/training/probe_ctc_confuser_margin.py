"""Compare complete phone hypotheses on existing trigger-conditioned development audio."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import itertools
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from stream_data import features, render_stream
from probe_causal_phone_verifier import EXTRA, LOOKBACK, MODEL, sha

COHORT = '3ea6c08ddbe344b285b66a498e13fe0e9fd74660cb0d5b903b027b8604394498'
FEATURES = 'a0971b3e513ce80bb800ccb55f39215008a2222e462ac0ba43f997f86374c283'
NEGATIVES = frozenset(('hey mix', 'hey next', 'hey nick'))
POSITIVES = frozenset(('hey nyx', 'hey nix', 'hey nicks'))
CANONICAL = {
    'nyx': [('h', 'eɪ', 'n', 'ɪ', 'k', 's')],
    'mix': [('h', 'eɪ', 'm', 'ɪ', 'k', 's')],
    'next': [('h', 'eɪ', 'n', 'ɛ', 'k', 's', 't')],
    'nick': [('h', 'eɪ', 'n', 'ɪ', 'k')],
}
# Secondary readout tolerates common phone-model vowel/nasal variation.
# The same shared Hey variants and equal per-class template priors are used.
VARIANTS = {
    name: sorted(set(('h', hey, onset, vowel, 'k') + tail
                    for hey, onset, vowel in itertools.product(
                        ('eɪ', 'eː', 'e', 'aɪ', 'ɛ'), ('m',) if name == 'mix' else ('n', 'ŋ'),
                        ('ɛ', 'e') if name == 'next' else ('ɪ', 'i'))
                    for tail in ({'nyx': [('s',)], 'mix': [('s',)],
                                  'next': [('s', 't')], 'nick': [()]}[name])))
    for name in CANONICAL
}


def phone_scores(logits, vocab, blank, templates):
    import torch
    phones = [tuple(vocab[token] for token in row) for rows in templates.values() for row in rows]
    targets = torch.zeros((len(phones), max(map(len, phones))), dtype=torch.long)
    for i, tokens in enumerate(phones):
        targets[i, :len(tokens)] = torch.tensor(tokens)
    log_probs = logits.log_softmax(-1)
    losses = torch.nn.functional.ctc_loss(
        log_probs[:, None, :].expand(-1, len(phones), -1).contiguous(), targets,
        torch.full((len(phones),), len(logits), dtype=torch.long),
        torch.tensor(list(map(len, phones))), blank=blank, reduction='none', zero_infinity=False)
    at, result = 0, {}
    for name, rows in templates.items():
        result[name] = float(torch.logsumexp(-losses[at:at+len(rows)], 0)-math.log(len(rows)))
        at += len(rows)
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite CTC hypothesis score')
    return {'log_probability': result, 'winner': max(result, key=result.get),
            'nyx_margin': result['nyx']-max(v for k, v in result.items() if k != 'nyx')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'cohort', 'backgrounds', 'phone-model', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report')
    plan = json.loads((Path(__file__).parent/'plans/microwake-near-word-errors-v1.json').read_bytes())
    if (sha(a.cohort) != COHORT or sha(a.features/'features.json') != FEATURES
            or sha(a.backgrounds/'manifest.json') != plan['background_manifest_sha256']
            or any(sha(a.phone_model/k) != v for k, v in plan['phoneme_model_files'].items())
            or any(version(k) != v for k, v in plan['runtime_packages'].items())):
        raise ValueError('Expected frozen cohort, rendering inputs and pinned phoneme model runtime')
    cohort = json.loads(a.cohort.read_bytes())
    if cohort['wake_model_sha256'] != MODEL or cohort['threshold'] != .994:
        raise ValueError('Changed trigger-conditioned model policy')
    selection = [r for r in cohort['rows'] if r['text'].replace(',', '').strip().lower().rstrip('.!?')
                 in (POSITIVES if r['label'] else NEGATIVES)]
    index = json.loads((a.features/'features.json').read_bytes())
    def source_key(row):
        return (row.get('source_sha256', row.get('sha256')), row['variant'], row['text'], row['engine'], row['label'])
    by_key = {source_key(r): r for r in index if r['split'] == 'validation'}
    if len(by_key) != sum(r['split'] == 'validation' for r in index):
        raise ValueError('Ambiguous source alias; do not infer a rendering from waveform hash alone')
    reserved = {r['sha256'] for r in index if r['split'] != 'validation'}
    bg = [str(a.backgrounds/r['path']) for r in json.loads((a.backgrounds/'manifest.json').read_bytes())['records']
          if r['split'] == 'validation']
    dataset = a.features/'dataset'
    generation = json.loads((dataset/'generation.json').read_bytes())
    cache = np.memmap(a.features/'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    model = Wav2Vec2ForCTC.from_pretrained(a.phone_model, local_files_only=True, weights_only=True).eval()
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(a.phone_model, local_files_only=True)
    vocab = json.loads((a.phone_model/'vocab.json').read_bytes())
    blank = model.config.pad_token_id
    originals = {}

    @torch.inference_mode()
    def score(pcm):
        value = extractor(pcm.astype(np.float32)/32768, sampling_rate=16000, return_tensors='pt')
        logits = model(**value).logits[0]
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite phone-model output')
        return {name: phone_scores(logits, vocab, blank, templates)
                for name, templates in (('canonical', CANONICAL), ('variants', VARIANTS))}

    results, counts = [], defaultdict(Counter)
    for n, old in enumerate(selection, 1):
        r = by_key[source_key(old)]
        path = Path(r['path'])
        if (path.is_absolute() or '..' in path.parts or r['sha256'] in reserved
                or (r['label'], r['text'], r['engine']) != (old['label'], old['text'], old['engine'])
                or sha(dataset/path) != r['sha256']):
            raise ValueError('Changed or reserved source')
        source, rate = sf.read(dataset/path, dtype='int16')
        if rate != 16000 or source.shape != (r['samples'],):
            raise ValueError('Expected unchanged 16 kHz mono PCM')
        seed = int(hashlib.sha256(r['path'].encode()).hexdigest()[:8], 16)+r['variant']
        wave, lead = render_stream(r, dataset, np.random.default_rng(seed), r['variant'], bg, generation['lead_seconds'])
        if lead//160 != r['lead_frame'] or not np.array_equal(features(wave), cache[r['offset']:r['offset']+r['frames']]):
            raise ValueError('Render does not reproduce the exact cached detector input')
        wave = np.rint(wave).astype(np.int16)
        at = old['trigger_sample']
        if not 0 < at <= len(wave):
            raise ValueError('Invalid original trigger clock')
        key = r['sha256']
        if key not in originals:
            originals[key] = score(source)
        views = {'full_source': originals[key], 'at_trigger': score(wave[max(0, at-LOOKBACK):at]),
                 'plus_240ms': score(wave[max(0, at-LOOKBACK):min(len(wave), at+EXTRA)])}
        label = 'nyx' if r['label'] else r['text'].strip().lower().rstrip('.!?').split()[-1]
        for view, policies in views.items():
            for policy, measured in policies.items():
                counts[f'{label}/{view}/{policy}'].update({
                    'views': 1, 'nyx_winner': int(measured['nyx_margin'] > 0),
                    'intended_class_winner': int(measured['winner'] == label)})
        results.append({**old, 'source': r['path'], 'intended_class': label, 'ctc_hypotheses': views})
        if n % 20 == 0 or n == len(selection):
            print(f'{n}/{len(selection)}', flush=True)
    report = {'scope': 'Standalone synthetic development clips, selected by an existing wake trigger. Diagnostic only.',
              'cohort_sha256': COHORT, 'wake_model_sha256': MODEL, 'threshold': .994,
              'source_sha256': sha(Path(__file__)),
              'feature_index_sha256': sha(a.features/'features.json'),
              'lookback_samples': LOOKBACK, 'extra_samples': EXTRA,
              'hypotheses': {'canonical': CANONICAL, 'variants': VARIANTS},
              'margin_rule': 'log mean P(Nyx phones) minus best competing class log mean probability; fixed margin zero',
              'counts': counts, 'rows': results,
              'limits': 'Relative forced phone likelihood is not a pronunciation label, deployable verifier, or human/acoustic acceptance. No fitting, waveform-speed or detector changes.'}
    with a.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
