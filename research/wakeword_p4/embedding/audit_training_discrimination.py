"""Measure a frozen training-only sample; never select thresholds or fit a model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from evaluate_equal_mean import EqualMeanHead
from evaluate_head import events
from train_head import MIN_END


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root.resolve()
    plan = json.loads(args.plan.read_text())
    if args.output.exists():
        parser.error('Use a fresh output directory')
    for file, expected in plan['helpers'].items():
        if sha(Path(__file__).with_name(file)) != expected:
            raise ValueError('Audit helper differs from the frozen plan')
    metadata_path = base / plan['source_text_metadata']
    if sha(metadata_path) != plan['source_text_metadata_sha256']:
        raise ValueError('Source text metadata differs')
    metadata = defaultdict(list)
    for row in json.loads(metadata_path.read_text()):
        if row['variant'] == 0:
            metadata[row['sha256']].append(row)
    jobs_path = base / 'head-features-tts-v1/jobs-private.json'
    if sha(jobs_path) != plan['source_jobs_sha256']:
        raise ValueError('Source jobs differ')
    jobs = {j['id']: j for j in json.loads(jobs_path.read_text())}
    chosen = {s['source_sha256']: s for s in plan['sources']}
    if len(chosen) != len(plan['sources']):
        raise ValueError('Repeated source in frozen sample')
    selected, heldout = [], set()
    for directory, binding in plan['feature_inputs'].items():
        for file, key in [('index.jsonl', 'index_sha256'), ('complete.json', 'complete_sha256')]:
            if sha(base / directory / file) != binding[key]:
                raise ValueError('Feature input differs')
        for line in (base / directory / 'index.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['split'] != 'training':
                heldout.add(row['source_sha256'])
            if row['source_sha256'] in chosen:
                if row['split'] != 'training' or row['kind'] != 'tts':
                    raise ValueError('Only training TTS rows are allowed')
                expected = chosen[row['source_sha256']]
                if (row['engine'], row['label']) != (expected['engine'], expected['label']):
                    raise ValueError('Source role differs')
                selected.append(row)
    if heldout.intersection(chosen):
        raise ValueError('Held-out source overlaps selected training')
    variants = defaultdict(set)
    for row in selected:
        variants[row['source_sha256']].add(row['variant'])
    if len(selected) != len(chosen) * 4 or any(v != {0, 1, 2, 3} for v in variants.values()):
        raise ValueError('Expected four original/environment views per source')
    heads = []
    for member in plan['members']:
        path = base / member['checkpoint'] / 'head.tflite'
        if sha(path) != member['head_sha256']:
            raise ValueError('Frozen original head differs')
        heads.append(path)
    detector = EqualMeanHead(heads)
    args.output.mkdir(parents=True)
    activity, texts = {}, {}
    for source, chosen_row in chosen.items():
        job = jobs[chosen_row['variant0_id']]
        if job['split'] != 'training' or job['source_sha256'] != source:
            raise ValueError('Source job is not the frozen training sample')
        entries = metadata[source]
        if not entries or any(e['split'] != 'training' or e['label'] != job['label'] for e in entries):
            raise ValueError('Source text role differs')
        unique_texts = {e['text'] for e in entries}
        if len(unique_texts) != 1:
            raise ValueError('Ambiguous source text metadata')
        texts[source] = next(iter(unique_texts))
        source_path = Path(job['source']).resolve()
        if not source_path.is_relative_to(metadata_path.resolve().parent / 'dataset'):
            raise ValueError('Source outside the verified synthetic dataset')
        if sha(source_path) != source:
            raise ValueError('Source waveform differs')
        pcm, rate = sf.read(source_path, dtype='int16')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) == 0:
            raise ValueError('Unexpected source format')
        padded = np.pad(pcm.astype(np.float64), (0, (-len(pcm)) % 160))
        rms = np.sqrt(np.mean(padded.reshape(-1, 160) ** 2, axis=1))
        active = np.flatnonzero(rms >= max(float(rms.max()) * .01, 1.))
        if not len(active):
            raise ValueError('Selected source has no activity')
        activity[source] = (2.4 + active[0] * .01, 2.4 + (active[-1] + 1) * .01 + .32)
    details = []
    for number, row in enumerate(selected, 1):
        path = Path(row['path'])
        path = path if path.is_absolute() else base / path
        if sha(path) != row['features_sha256']:
            raise ValueError('Feature payload differs')
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.int8 or array.shape != (row['frames'], 96):
            raise ValueError('Unexpected feature format')
        scores, times = detector.scores(array, plan['smoothing_steps'])
        valid = ((times >= row['keyword_end'] - .1) & (times <= row['keyword_end'] + .8)) if row['label'] else np.ones(len(times), bool)
        maximum = float(scores[valid].max()) if valid.any() else 0.
        record = {key: row[key] for key in ('id', 'engine', 'label', 'variant', 'source_sha256', 'features_sha256')}
        record.update({'text': texts[row['source_sha256']], 'max_score': maximum,
            'hit_or_false_clip': maximum >= plan['threshold'], 'events': events(scores, plan['threshold'])})
        if row['label']:
            record['premature_threshold_crossing'] = bool(np.any((times < row['keyword_end'] - .1) & (scores >= plan['threshold'])))
        else:
            ends = np.arange(MIN_END, row['frames'] + 1) * .08
            start, stop = activity[row['source_sha256']]
            record['uniform_endpoint_source_activity_fraction'] = float(np.mean((ends >= start) & (ends <= stop)))
        details.append(record)
        if number % 256 == 0 or number == len(selected):
            print(json.dumps({'scored_training_rows': number, 'total': len(selected)}), flush=True)
    groups = defaultdict(list)
    for row in details:
        groups[(row['engine'], row['label'])].append(row)
    cohorts = []
    for (engine, label), rows in sorted(groups.items()):
        scores = [r['max_score'] for r in rows]
        entry = {'engine': engine, 'label': label, 'sources': len({r['source_sha256'] for r in rows}),
            'variants': len(rows), 'threshold_clips': sum(r['hit_or_false_clip'] for r in rows),
            'score_quantiles_05_50_95': np.quantile(scores, [.05, .5, .95]).tolist()}
        key = 'premature_threshold_crossing' if label else 'uniform_endpoint_source_activity_fraction'
        entry[key + '_mean'] = float(np.mean([r[key] for r in rows]))
        cohorts.append(entry)
    phrases = defaultdict(list)
    for row in details:
        if not row['label']:
            phrases[row['text']].append(row)
    result = {'scope': __doc__, 'plan_sha256': sha(args.plan), 'sources': len(chosen), 'variants': len(details),
        'all_inputs_training_only': True, 'heldout_source_hash_disjoint': True,
        'threshold': plan['threshold'], 'cohorts': cohorts,
        'negative_phrases': [{'text': text, 'variants': len(rows), 'false_clips': sum(r['hit_or_false_clip'] for r in rows)}
            for text, rows in sorted(phrases.items())],
        'accepted': False, 'limits': 'In-sample descriptive evidence only. Source activity endpoints omit the full embedding receptive field. TTS text/RMS boundaries are not phonetic ground truth; no detector threshold or new training sample is chosen from these scores.'}
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    (args.output / 'rows-private.json').write_text(json.dumps(details, indent=2) + '\n')
    print(json.dumps({'cohorts': cohorts}), flush=True)


if __name__ == '__main__':
    main()
