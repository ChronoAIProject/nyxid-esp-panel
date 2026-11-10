"""Measure fixed-head sensitivity to sample alignment on training sources only.

Shift silence from the tail to the lead without changing speech samples,
recording length, sample rate or speed. This is a diagnostic, not wake accuracy
acceptance or a calibration/threshold search.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import prepare_head_features as extractor
from evaluate_equal_mean import EqualMeanHead

HEAD = None
OFFSETS = None
THRESHOLD = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(root, plan):
    global HEAD, OFFSETS, THRESHOLD
    extractor.initialize(root / plan['backbone'], root / plan['window'], root / plan['mel_filter'])
    HEAD = EqualMeanHead([root / row['path'] for row in plan['heads']])
    OFFSETS, THRESHOLD = plan['offset_samples'], plan['threshold']


def worker(job):
    if digest(job['source']) != job['source_sha256'] or digest(job['features']) != job['features_sha256']:
        raise ValueError('Training source or original feature identity differs')
    pcm, rate = sf.read(job['source'], dtype='int16')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Expected the unchanged mono16kHz training recording')
    original = np.load(job['features'], allow_pickle=False)
    phases = []
    for offset in OFFSETS:
        signal = np.pad(pcm, (38400 + offset, 16000 - offset))
        if len(signal) != len(pcm) + 54400 or not np.array_equal(signal[38400 + offset:38400 + offset + len(pcm)], pcm):
            raise ValueError('Speech samples or duration changed')
        embedding = extractor.extract(signal)
        if offset == 0 and not np.array_equal(embedding, original):
            raise ValueError('Phase-zero extraction does not reproduce original training features')
        scores, times = HEAD.scores(embedding, 3)
        if job['label']:
            end = (38400 + offset + job['keyword_end']) / 16000
            scores = scores[(times >= end - .1) & (times <= end + .8)]
        if not len(scores):
            raise ValueError('Empty score interval')
        peak = float(scores.max())
        phases.append({'offset_samples': offset, 'peak': peak, 'above_frozen_threshold': peak >= THRESHOLD})
    return {'id': job['id'], 'engine': job['engine'], 'label': job['label'],
            'speech_sha256': hashlib.sha256(pcm.tobytes()).hexdigest(),
            'phase_zero_features_exact': True, 'phases': phases}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if digest(Path(__file__)) != plan['audit_script_sha256']:
        raise ValueError('Audit implementation differs from the frozen plan')
    if not np.isfinite(plan['threshold']) or not 0 < plan['threshold'] < 1:
        raise ValueError('Expected a frozen finite probability threshold')
    if args.workers < 1 or type(plan['per_engine_label']) is not int or plan['per_engine_label'] < 1:
        p.error('workers and sample counts must be positive integers')
    offsets = plan['offset_samples']
    if not offsets or offsets[0] != 0 or len(set(offsets)) != len(offsets) or any(type(n) is not int or not 0 <= n < 1280 for n in offsets):
        raise ValueError('Offsets must include zero and stay inside the original80ms sample block')
    for name, expected in plan['input_files'].items():
        if digest(args.data_root / name) != expected:
            raise ValueError('Frozen audit input identity differs')
    for row in plan['heads']:
        if digest(args.data_root / row['path']) != row['sha256']:
            raise ValueError('Frozen head identity differs')
    rows = json.loads((args.data_root / plan['jobs']).read_text())
    heldout = {row['source_sha256'] for row in rows if row['split'] != 'training'}
    features = {row['id']: row for row in map(json.loads, (args.data_root / plan['index']).read_text().splitlines())}
    groups = {}
    for row in rows:
        if row['split'] == 'training' and row['variant'] == 0 and row['kind'] == 'tts':
            if row['source_sha256'] in heldout:
                raise ValueError('Training and held-out source identities overlap')
            groups.setdefault((row['engine'], int(row['label'])), []).append(row)
    rng = np.random.default_rng(plan['seed'])
    jobs = []
    for _, group in sorted(groups.items()):
        group.sort(key=lambda row: row['id'])
        for at in rng.choice(len(group), min(plan['per_engine_label'], len(group)), replace=False):
            row = dict(group[int(at)])
            feature = features[row['id']]
            if feature['split'] != 'training' or feature['source_sha256'] != row['source_sha256']:
                raise ValueError('Source and feature training identities differ')
            path = (args.data_root / feature['path']).resolve()
            if not path.is_relative_to(args.data_root.resolve()):
                raise ValueError('Training features are outside the private data root')
            row.update(features=str(path), features_sha256=feature['features_sha256'])
            jobs.append(row)
    if not jobs:
        raise ValueError('Empty training source selection')
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.data_root, plan)) as pool:
        results = list(pool.map(worker, jobs))
    positives = [row for row in results if row['label']]
    summary = {'scope': __doc__, 'plan_sha256': digest(args.plan), 'sources': len(results),
        'positive_sources': len(positives), 'all_phase_zero_features_exact': True,
        'positive_threshold_outcome_varies_by_phase': sum(len({v['above_frozen_threshold'] for v in row['phases']}) > 1 for row in positives),
        'positive_above_threshold_by_offset': {str(offset): sum(row['phases'][at]['above_frozen_threshold'] for row in positives)
            for at, offset in enumerate(offsets)},
        'heldout_sources_read': False, 'audio_speed_changed': False, 'accepted': False}
    (args.output / 'rows-private.json').write_text(json.dumps(results, indent=2) + '\n')
    (args.output / 'report.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
