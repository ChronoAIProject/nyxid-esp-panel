"""Audit fixed training-source markers with an unprompted phoneme recognizer.

CTC feature-frame support is a diagnostic, not a phonetic boundary label.
No wake detector is loaded, no source is relabelled and no audio is played.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
import wave


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def feature_geometry(kernels, strides):
    if len(kernels) != len(strides) or not kernels:
        raise ValueError('Invalid feature convolution geometry')
    field, step = 1, 1
    for kernel, stride in zip(kernels, strides):
        if type(kernel) is not int or type(stride) is not int or min(kernel, stride) < 1:
            raise ValueError('Invalid feature convolution dimensions')
        field += (kernel - 1) * step
        step *= stride
    return step, field


def token_spans(ids, confidences, tokens, blank, step, field, samples):
    if len(ids) != len(confidences):
        raise ValueError('Frame/confidence lengths differ')
    result = []
    start = 0
    while start < len(ids):
        stop = start + 1
        while stop < len(ids) and ids[stop] == ids[start]:
            stop += 1
        token = ids[start]
        if token != blank:
            result.append({
                'token': tokens[token], 'frame_start': start, 'frame_stop': stop,
                'feature_input_support_samples': [
                    start * step, min(samples, (stop - 1) * step + field)],
                'mean_argmax_probability': sum(confidences[start:stop]) / (stop - start),
            })
        start = stop
    return result


def standalone(text):
    return bool(re.fullmatch(r'hey[\s,]+(?:nyx|nix|nicks)[.!?,]*', text.strip(), re.I))


def load_sources(base, parent_path, plan):
    if sha(parent_path) != plan['parent_plan_sha256']:
        raise ValueError('Parent selection plan differs')
    parent = json.loads(parent_path.read_text())
    jobs_path = base / 'head-features-tts-v1/jobs-private.json'
    metadata_path = (base / parent['source_text_metadata']).resolve()
    if sha(jobs_path) != parent['source_jobs_sha256']:
        raise ValueError('Source jobs differ')
    if sha(metadata_path) != parent['source_text_metadata_sha256']:
        raise ValueError('Source metadata differs')
    jobs = json.loads(jobs_path.read_text())
    by_id = {job['id']: job for job in jobs}
    if len(by_id) != len(jobs):
        raise ValueError('Duplicate source jobs')
    heldout = {j['source_sha256'] for j in jobs if j['split'] != 'training'}
    metadata = defaultdict(list)
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] == 0:
            path = (metadata_path.parent / 'dataset' / row['path']).resolve()
            metadata[path].append(row)
    for directory, binding in parent['feature_inputs'].items():
        for name, key in [('index.jsonl', 'index_sha256'), ('complete.json', 'complete_sha256')]:
            if sha(base / directory / name) != binding[key]:
                raise ValueError('Parent feature identity differs')
        for line in (base / directory / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'training':
                heldout.add(row['source_sha256'])
    selected, omitted = [], []
    for source in parent['sources']:
        if source['label'] != 1:
            continue
        job = by_id[source['variant0_id']]
        path = Path(job['source']).resolve()
        rows = metadata[path]
        if (job['split'], job['label'], job['kind'], job['variant']) != ('training', 1, 'tts', 0):
            raise ValueError('Selected source is not an original training positive')
        if (job['source_sha256'], job['engine']) != (source['source_sha256'], source['engine']) or source['source_sha256'] in heldout:
            raise ValueError('Selected source identity or split differs')
        if not rows or any(
            (r['split'], r['label'], r['sha256'], r['engine'], r['keyword_end']) !=
            ('training', 1, job['source_sha256'], job['engine'], job['keyword_end'])
            for r in rows
        ):
            raise ValueError('Source metadata role/hash/marker differs')
        texts = sorted({r['text'] for r in rows})
        identity = {k: source[k] for k in ('source_sha256', 'variant0_id', 'engine')}
        identity.update(texts=texts, keyword_end=job['keyword_end'])
        if not all(standalone(text) for text in texts):
            omitted.append(identity)
            continue
        if sha(path) != source['source_sha256']:
            raise ValueError('Source waveform differs')
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
                raise ValueError('Expected unchanged 16 kHz mono PCM16 source')
            identity['samples'] = wav.getnframes()
        selected.append((identity, path))
    if [item for item, _ in selected] != plan['sources'] or omitted != plan['omitted_command_sources']:
        raise ValueError('Standalone selection differs from frozen plan')
    if len({item['source_sha256'] for item, _ in selected}) != len(selected):
        raise ValueError('Duplicate selected audio')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--parent-plan', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    plan = json.loads(args.plan.read_text())
    if sha(Path(__file__)) != plan['script_sha256']:
        raise ValueError('Audit script differs from frozen plan')
    for name, expected in plan['phoneme_model_files'].items():
        if sha(args.model / name) != expected:
            raise ValueError('Independent phoneme model file differs: ' + name)
    runtime = {name: version(name) for name in plan['runtime_packages']}
    if runtime != plan['runtime_packages']:
        raise ValueError('Pinned diagnostic runtime differs')
    selected = load_sources(args.data_root.resolve(), args.parent_plan, plan)

    # Heavy model imports happen only after role/hash checks; no expected text
    # is supplied to the recognizer and no mutable remote model is downloaded.
    import numpy as np
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(args.model, local_files_only=True, weights_only=True)
    model.eval()
    tokens = {value: key for key, value in json.loads((args.model / 'vocab.json').read_text()).items()}
    step, field = feature_geometry(model.config.conv_kernel, model.config.conv_stride)
    if (step, field) != (plan['feature_stride_samples'], plan['feature_support_samples']):
        raise ValueError('Feature geometry differs from frozen plan')
    args.output.mkdir(parents=True)
    results = []
    started = time.monotonic()
    with (args.output / 'rows.jsonl').open('x') as log, torch.inference_mode():
        for index, (identity, path) in enumerate(selected, 1):
            if sha(path) != identity['source_sha256']:
                raise ValueError('Source changed after preflight')
            with wave.open(str(path), 'rb') as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2')
            inputs = extractor(pcm.astype(np.float32) / 32768, sampling_rate=16000, return_tensors='pt')
            logits = model(**inputs).logits[0]
            expected_frames = (len(pcm) - field) // step + 1
            if len(logits) != expected_frames:
                raise ValueError('Actual output frame count differs from feature geometry')
            probabilities, ids = logits.softmax(dim=-1).max(dim=-1)
            spans = token_spans(ids.tolist(), probabilities.tolist(), tokens,
                                model.config.pad_token_id, step, field, len(pcm))
            padded = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
            rms = np.sqrt(np.mean(padded.reshape(-1, 160) ** 2, axis=1))
            active = np.flatnonzero(rms >= max(float(rms.max()) * .01, 1.))
            activity_end = min(len(pcm), int((active[-1] + 1) * 160)) if len(active) else None
            support_end = spans[-1]['feature_input_support_samples'][1] if spans else None
            row = dict(identity, phonemes=[s['token'] for s in spans], spans=spans,
                       activity_end_samples=activity_end,
                       last_ctc_feature_support_end_samples=support_end,
                       marker_minus_activity_end_ms=(identity['keyword_end'] - activity_end) / 16 if activity_end is not None else None,
                       marker_minus_ctc_support_end_ms=(identity['keyword_end'] - support_end) / 16 if support_end is not None else None)
            results.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + '\n')
            log.flush()
            if index % 12 == 0 or index == len(selected):
                print(json.dumps({'completed': index, 'total': len(selected), 'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    groups = defaultdict(list)
    for row in results:
        groups[row['engine']].append(row)
    cohorts = []
    for engine, rows in sorted(groups.items()):
        cohort = {'engine': engine, 'sources': len(rows), 'all_blank': sum(not r['spans'] for r in rows),
                  'terminal_token_counts': dict(Counter(r['spans'][-1]['token'] for r in rows if r['spans']))}
        for key in ('marker_minus_activity_end_ms', 'marker_minus_ctc_support_end_ms'):
            values = [r[key] for r in rows if r[key] is not None]
            cohort[key] = {'available': len(values),
                           'quantiles_05_50_95': np.quantile(values, [.05, .5, .95]).tolist() if values else [],
                           'over_200ms': sum(v > 200 for v in values), 'over_500ms': sum(v > 500 for v in values)}
        cohorts.append(cohort)
    report = {
        'scope': __doc__, 'plan_sha256': sha(args.plan), 'script_sha256': sha(Path(__file__)),
        'sources': len(results), 'omitted_command_sources': len(plan['omitted_command_sources']),
        'rows_sha256': sha(args.output / 'rows.jsonl'), 'phoneme_model_files': plan['phoneme_model_files'],
        'runtime_packages': runtime,
        'feature_stride_samples': step, 'feature_support_samples': field,
        'cohorts': cohorts, 'original_audio_and_labels_changed': False,
        'heldout_audio_loaded': False, 'wake_detector_loaded': False, 'audio_played': False,
        'accepted': False,
        'limits': [
            'Unprompted learned phoneme predictions and local feature-frame support are not phonetic ground truth or exact token acoustic boundaries.',
            'The transformer uses full utterance context; the reported convolution support is not the entire model receptive field.',
            'CTC spikes and RMS endpoints are independent diagnostics. Neither automatically replaces keyword labels or establishes wake quality.',
            'This fixed training-only standalone sample omits command-followed aliases; no validation/test scoring or detector threshold fitting occurs.',
        ],
    }
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'cohorts': cohorts}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
