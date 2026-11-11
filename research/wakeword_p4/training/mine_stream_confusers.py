"""Screen continuous, training-only LibriSpeech for streaming detector confusers."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import Detector, events, smooth

THRESHOLDS = (.5, .9, .97)
SMOOTH = 5
MODELS = {
    'phrase_control': 'c8c7e66b6f212f2b18c1e26c9f3bd27176fc0bd249361447fc641ac1d3f83ce5',
    'phrase_aux': '1919dcf10e4d422c2420d6cdda5a718813fe5044c759bc2dea58807382330af7',
}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def peaks(scores):
    # Consecutive high-score groups are evidence of difficult audio, even if
    # the listener's refractory rule would merge multiple adjacent groups.
    high = np.flatnonzero(scores >= THRESHOLDS[0])
    groups = np.split(high, np.flatnonzero(np.diff(high) > 1)+1) if len(high) else []
    return [{'first_sample': (3*(int(g[0])+SMOOTH-1)+5)*160,
             'peak_sample': (3*(int(g[np.argmax(scores[g])])+SMOOTH-1)+5)*160,
             'peak_score': float(scores[g].max())} for g in groups]


def measure(job):
    root, row, models = job
    path = root/'features'/f"{row['speaker']}.npy"
    if sha(path) != row['feature_sha256']:
        raise ValueError('Original training feature checksum mismatch')
    raw = np.load(path, mmap_mode='r')
    if raw.shape != (row['frames'], 40) or raw.dtype != np.uint16:
        raise ValueError('Original training feature geometry differs')
    reports = {}
    for name, model in models.items():
        values = smooth(Detector(model).predict(raw), SMOOTH)
        reports[name] = {'maximum_score': float(values.max()),
                         'events': {str(t): events(values, t) for t in THRESHOLDS},
                         'high_score_groups': peaks(values)}
    return {'speaker': row['speaker'], 'seconds': row['seconds'], 'models': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'existing-index', 'experiment', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report')
    manifest = args.training/'manifest.json'
    data = json.loads(manifest.read_bytes())
    speakers = set(data['speaker_partitions']['training'])
    reserved = {r.get('speaker') for r in json.loads(args.existing_index.read_bytes()) if r['split'] != 'training'}
    rows = data['records']
    if (data['archive_sha256'] != 'd4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2'
            or len(speakers) != 155 or speakers & reserved
            or data['speaker_partitions']['validation'] or data['speaker_partitions']['testing']
            or len(rows) != len(speakers) or {r['speaker'] for r in rows} != speakers
            or any(r['split'] != 'training' or not r['speaker'].isdigit() for r in rows)):
        raise ValueError('Expected the disjoint training-only public speech extension')
    models = {name: args.experiment/f'model-{name}'/'hey_nyx.tflite' for name in MODELS}
    if {name: sha(path) for name, path in models.items()} != MODELS:
        raise ValueError('Use the frozen paired phrase artifacts')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(measure, [(args.training, row, models) for row in rows]):
            results.append(r)
            if len(results) % 20 == 0 or len(results) == len(rows):
                print(json.dumps({'scored_speakers': len(results), 'total_speakers': len(rows)}), flush=True)
    totals = {}
    for name in models:
        groups = [r['models'][name] for r in results]
        tally = Counter()
        for row in groups:
            tally.update(row['events'])
        totals[name] = {'model_sha256': MODELS[name], 'events': dict(tally),
                        'max_score': max(r['maximum_score'] for r in groups),
                        'speakers_with_confusers': sum(bool(r['high_score_groups']) for r in groups)}
    report = {'scope': 'Training-only hard negative mining, not acceptance',
              'source_manifest_sha256': sha(manifest), 'partition_index_sha256': sha(args.existing_index),
              'speaker_count': len(results), 'hours': sum(r['seconds'] for r in results)/3600,
              'thresholds': THRESHOLDS, 'models': totals, 'streams': results}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'streams'}), flush=True)


if __name__ == '__main__':
    main()
