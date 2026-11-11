"""Check whether text-mined training spans are harder than matched ordinary audio."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluate import Detector, smooth

THRESHOLDS = (.5, .9, .97)
HISTORY = 400  # Longer than the reviewed 335-frame model's receptive field.
WIDTH = 20  # 600 ms of score samples, beginning 150 ms before the estimated end.


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def match_window(row, frame_count):
    origin = row['origin_samples']
    a = max(0, (origin // 160 - HISTORY) // 3 * 3)
    b = min(frame_count, (origin + row['source_samples']) // 160)
    ends = a*160 + (3*(np.arange((b-a)//3-4) + 4) + 5)*160
    forbidden = [(origin + s['onset_samples'] - 2400,
                  origin + s['end_samples'] + 7200 + HISTORY*160) for s in row['spans']]
    options = [i for i in np.flatnonzero(ends >= origin+4800) if i+WIDTH <= len(ends)
               and all(ends[i+WIDTH-1] < lo or ends[i] > hi for lo, hi in forbidden)]
    if not options:
        return None
    ordinary = options[int.from_bytes(hashlib.sha256(row['source'].encode()).digest()[:8], 'big') % len(options)]
    for span in sorted(row['spans'], key=lambda s: (s['distance'], -len(s['phones']), s['end_samples'])):
        near = int(np.searchsorted(ends, origin + span['end_samples'] - 2400))
        if near+WIDTH <= len(ends):
            return (a, b, slice(near, near+WIDTH), slice(ordinary, ordinary+WIDTH), ends)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('aligned', 'features', 'experiment', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    aligned = json.loads(args.aligned.read_bytes())
    index = args.features/'human.json'
    if sha(index) != aligned['index_sha256']:
        raise ValueError('Original training index required')
    rows = json.loads(index.read_bytes())
    streams = {r['path']: r for r in rows if r['split'] == 'training'}
    grouped = defaultdict(list)
    for row in aligned['rows']:
        stream = streams[row['feature_path']]
        at = 0
        for source in stream['sources']:
            if source['utterance'] == row['source']:
                break
            at += source['samples'] + 3200
        else:
            raise ValueError('Missing utterance in the training stream')
        if (stream['speaker'] != row['speaker'] or stream['feature_sha256'] != row['feature_sha256']
                or (row['source_sha256'], row['source_samples'], row['origin_samples']) !=
                    (source['flac_sha256'], source['samples'], at)):
            raise ValueError('Speech slice is not from the original training stream')
        grouped[row['feature_path']].append(row)
    models = {}
    for name in ('phrase_control', 'phrase_aux'):
        path = args.experiment/f'model-{name}'/'hey_nyx.tflite'
        info = json.loads((path.parent/'training.json').read_bytes())
        if sha(path) != info['model_sha256'] or info['input_frames'] != 335:
            raise ValueError('Expected the frozen paired-phrase experiment')
        models[name] = Detector(path)
    pairs = []
    for path, sources in grouped.items():
        file = args.features/path
        if sha(file) != streams[path]['feature_sha256']:
            raise ValueError('Training feature array checksum mismatch')
        raw = np.load(file, mmap_mode='r')
        for row in sources:
            match = match_window(row, len(raw))
            if match is None:
                continue
            a, b, near, ordinary, ends = match
            pair = {'speaker': row['speaker'], 'source': row['source'],
                    'near_first_score_sample': int(ends[near.start]),
                    'ordinary_first_score_sample': int(ends[ordinary.start])}
            for name, detector in models.items():
                detector.reset()
                scores = smooth(detector.predict(raw[a:b]), 5)
                pair[name] = [float(scores[near].max()), float(scores[ordinary].max())]
            pairs.append(pair)
    if not pairs:
        raise ValueError('No matched windows')
    summary = {}
    for name, model in models.items():
        values = np.array([p[name] for p in pairs])
        summary[name] = {'model_sha256': sha(model.model),
                         'counts_above': {t: (values >= t).sum(axis=0).tolist() for t in THRESHOLDS},
                         'p95': np.quantile(values, .95, axis=0).tolist(),
                         'pairs_near_larger_tied_smaller': [int((values[:, 0] > values[:, 1]).sum()),
                             int((values[:, 0] == values[:, 1]).sum()), int((values[:, 0] < values[:, 1]).sum())]}
    report = {'scope': 'Training-only retrospective diagnostic; paired near/ordinary score windows, no calibration',
              'aligned_sha256': sha(args.aligned), 'matched_utterances': len(pairs),
              'matched_speakers': len({r['speaker'] for r in pairs}),
              'window_score_steps': WIDTH, 'column_order': ['near', 'ordinary'], 'models': summary,
              'limits': 'Dictionary candidates and ASR times are approximate. The control window excludes the matched spans, '
                        'but its words were not otherwise phonetically or acoustically reviewed.'}
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
