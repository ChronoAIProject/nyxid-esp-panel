"""Read-only phoneme diagnosis of two already scored synthetic validation phrases."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from compare_stream_equal_mean import checked
from stream_data import features, render_stream

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'embedding'))
from audit_ctc_endpoints import feature_geometry, sha, token_spans


def cohort(base, comparison, overlap, parent):
    names = list(comparison['members'])
    details = [json.loads(checked(base / f'validation-{n}/validation-details.json',
                                  comparison['members'][n]['validation_details_sha256']))['validation']
               for n in names]
    index = json.loads(checked(parent / 'features.json', comparison['feature_index_sha256']))
    other_splits = {r['sha256'] for r in index if r['split'] != 'validation'}
    entries = defaultdict(list)
    for row, a, b in zip((r for r in index if r['split'] == 'validation'), *details, strict=True):
        phrase = row['text'].lower().strip().rstrip('.!?')
        if phrase not in ('hey mix', 'hey next'):
            continue
        path = Path(row['path'])
        if row['label'] != 0 or path.is_absolute() or '..' in path.parts or row['sha256'] in other_splits:
            raise ValueError('Selection is not a source-disjoint validation negative')
        for side in (a, b):
            if side['source'] != row['path'] or any(side[k] != row[k] for k in ('engine', 'variant', 'label', 'text')):
                raise ValueError('Cached scores differ in source identity')
        x, y = (np.asarray(side['scores']) for side in (a, b))
        if (x.shape != (row['frames'] // 3 - 4,) or y.shape != x.shape or a['times'] != b['times']
                or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('Cached scores differ in geometry')
        thresholds = overlap['thresholds']
        entries[row['path']].append((row, phrase, {
            'strict': bool(((x + y) * .5 >= thresholds['strict_equal_mean']).any()),
            'both': bool((x >= thresholds[names[0]]).any() and (y >= thresholds[names[1]]).any()),
        }))
    for phrase in ('hey mix', 'hey next'):
        subset = [flags for views in entries.values() for _, p, flags in views if p == phrase]
        expected = overlap['negative'][phrase]
        if (len(subset), sum(f['strict'] for f in subset), sum(f['both'] for f in subset)) != (
                expected['clips'], expected['strict_equal_mean'], expected['both']):
            raise ValueError('Cached case counts differ from the published comparison')
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('artifacts', 'backgrounds', 'plan', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output path')
    plan = json.loads(args.plan.read_bytes())
    root = Path(__file__).resolve().parents[1]
    for name, expected in plan['research_source_sha256'].items():
        if sha(root / name) != expected:
            raise ValueError(f'Changed experiment source: {name}')
    comparison = json.loads(checked(root / 'training/plans/microwake-equal-mean-v1.json', plan['comparison_sha256']))
    overlap = json.loads(checked(root / 'results/microwake-error-overlap-v1.json', plan['overlap_sha256']))
    parent = args.artifacts / comparison['validation_feature_directory']
    entries = cohort(args.artifacts, comparison, overlap, parent)
    generation = json.loads(checked(parent / 'dataset/generation.json', plan['generation_sha256']))
    bg_rows = json.loads(checked(args.backgrounds / 'manifest.json', plan['background_manifest_sha256']))['records']
    backgrounds = [str(args.backgrounds / r['path']) for r in bg_rows if r['split'] == 'validation']
    if not backgrounds:
        raise ValueError('Missing original validation backgrounds')
    cache = np.memmap(parent / 'features.bin', dtype='<u2', mode='r').reshape(-1, 40)
    model_path = args.artifacts / 'phoneme-audit-model'
    for name, digest in plan['phoneme_model_files'].items():
        if sha(model_path / name) != digest:
            raise ValueError(f'Changed independent phoneme model file: {name}')
    if {name: version(name) for name in plan['runtime_packages']} != plan['runtime_packages']:
        raise ValueError('Pinned diagnostic runtime differs')

    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(model_path, local_files_only=True, weights_only=True).eval()
    tokens = {v: k for k, v in json.loads((model_path / 'vocab.json').read_bytes()).items()}
    step, field = feature_geometry(model.config.conv_kernel, model.config.conv_stride)
    if (step, field) != (320, 400):
        raise ValueError('Changed phoneme model geometry')

    @torch.inference_mode()
    def recognize(pcm):
        data = extractor(pcm.astype(np.float32) / 32768, sampling_rate=16000, return_tensors='pt')
        logits = model(**data).logits[0]
        if len(logits) != (len(pcm) - field) // step + 1 or not torch.isfinite(logits).all():
            raise ValueError('Invalid phoneme model output')
        conf, ids = logits.softmax(dim=-1).max(dim=-1)
        return token_spans(ids.tolist(), conf.tolist(), tokens, model.config.pad_token_id, step, field, len(pcm))

    results, counts, seen = [], defaultdict(Counter), set()
    for path, views in sorted(entries.items()):
        source = parent / 'dataset' / path
        identity, phrase, _ = views[0]
        digest = identity['sha256']
        if sha(source) != digest or digest in seen or sorted(r['variant'] for r, _, _ in views) != list(range(4)):
            raise ValueError('Changed/duplicated source audio or missing rendering')
        seen.add(digest)
        original, rate = sf.read(source, dtype='int16')
        if rate != 16000 or original.shape != (identity['samples'],) or sf.info(source).subtype != 'PCM_16':
            raise ValueError('Changed source PCM geometry')
        raw = recognize(original)
        rows = []
        for row, text, flags in sorted(views, key=lambda view: view[0]['variant']):
            if (row['sha256'], row['engine'], row['speakers'], text) != (
                    digest, identity['engine'], identity['speakers'], phrase):
                raise ValueError('Changed variant identity')
            variant = row['variant']
            seed = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) + variant
            pcm, lead = render_stream(row, parent / 'dataset', np.random.default_rng(seed),
                                      variant, backgrounds, generation['lead_seconds'])
            start = row['offset']
            if lead // 160 != row['lead_frame'] or not np.array_equal(features(pcm), cache[start:start + row['frames']]):
                raise ValueError(f'Render does not reproduce cached detector features: {path}, {variant}')
            # Preserve the whole phrase and 250 ms afterwards (including any
            # reverberation); avoid letting seconds of background dominate CTC.
            crop = [lead - 2560, lead + len(original) + 4000]
            spans = recognize(np.rint(pcm[slice(*crop)]).astype(np.int16))
            # Consonant presence is a descriptive readout, not relabelling.
            marker = {'hey mix': 'm', 'hey next': 't'}[phrase]
            def evidence(seq):
                return any(s['token'] == marker and s['mean_argmax_probability'] >= .5 for s in seq)

            for group, enabled in [('strict_fire', flags['strict']), ('strict_quiet', not flags['strict']),
                                   ('both_fire', flags['both'])]:
                if enabled:
                    counts[f'{phrase}/{group}'].update({'views': 1, 'raw_marker': evidence(raw),
                                                        'rendered_marker': evidence(spans)})
            rows.append({'variant': variant, **flags, 'crop_samples': crop,
                         'rendered_marker': evidence(spans), 'spans': spans})
        results.append({'source': path, 'source_sha256': digest, 'phrase': phrase, 'engine': identity['engine'],
                        'speakers': identity['speakers'], 'raw_marker': evidence(raw), 'raw': raw, 'views': rows})
        if len(results) % 12 == 0 or len(results) == len(entries):
            print(f'{len(results)}/{len(entries)} sources', flush=True)
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'counts': counts, 'rows': results}
    with args.output.open('x') as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
