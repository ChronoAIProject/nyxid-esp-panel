"""Add one normal-speed sample-phase view per original training TTS source.

Original sources, features and held-out partitions remain unchanged. Shift
silence between lead and tail; never stretch speech, crop it or change duration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf

import prepare_head_features as extractor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(job):
    if digest(job['source']) != job['source_sha256']:
        raise ValueError('Training source identity differs')
    pcm, rate = sf.read(job['source'], dtype='int16')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Expected original mono16kHz training PCM')
    offset = job['offset_samples']
    signal = np.pad(pcm, (38400 + offset, 16000 - offset))
    if len(signal) != len(pcm) + 54400 or not np.array_equal(signal[38400 + offset:38400 + offset + len(pcm)], pcm):
        raise ValueError('Speech samples or full duration changed')
    features = extractor.extract(signal)
    output = Path(job['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    row = {k: job[k] for k in ['id', 'split', 'kind', 'label', 'source_sha256', 'variant', 'engine', 'offset_samples']}
    row.update(path=str(output), frames=len(features), seconds=len(signal) / 16000,
        source_seconds=len(pcm) / 16000,
        keyword_end=(38400 + offset + job['keyword_end']) / 16000 if job['label'] else None,
        features_sha256=digest(output), pcm_sha256=hashlib.sha256(signal.tobytes()).hexdigest())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.workers < 1:
        p.error('workers must be positive')
    if digest(Path(__file__)) != plan['extractor_sha256']:
        raise ValueError('Phase extractor differs from frozen plan')
    for name, expected in plan['input_files'].items():
        if digest(args.data_root / name) != expected:
            raise ValueError('Frozen source/frontend/backbone identity differs')
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Feature implementation differs')
    if plan['offset_range_samples'] != [1, 1280]:
        raise ValueError('Expected the fixed within-block phase range')
    rows = json.loads((args.data_root / plan['jobs']).read_text())
    heldout = {r['source_sha256'] for r in rows if r['split'] != 'training'}
    source = sorted((r for r in rows if r['split'] == 'training' and r['kind'] == 'tts' and r['variant'] == 0), key=lambda r: r['id'])
    if not source or heldout.intersection(r['source_sha256'] for r in source):
        raise ValueError('Empty training pool or held-out source overlap')
    args.output = args.output.resolve()
    plan_hash = digest(args.plan)
    rng = np.random.default_rng(plan['seed'])
    jobs = []
    for row in source:
        identity = hashlib.sha256((row['id'] + ':phase:' + plan_hash).encode()).hexdigest()
        jobs.append({**row, 'id': identity, 'variant': 4,
            'offset_samples': int(rng.integers(1, 1280)),
            'output': str(args.output / 'training/tts' / (identity + '.npy'))})
    if len({r['id'] for r in jobs}) != len(jobs):
        raise ValueError('Duplicate phase-view identity')
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {'scope': __doc__, 'all_training_only': True, 'model_sha256': plan['input_files'][plan['backbone']],
        'plan_sha256': plan_hash, 'source_sha256': digest(Path(__file__)), 'jobs': len(jobs),
        'audio_speed_changed': False, 'full_recording_duration_changed': False}
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2) + '\n')
    (args.output / 'jobs-private.json').write_text(json.dumps(jobs) + '\n')
    started = time.monotonic()
    with (args.output / 'index.jsonl').open('x') as log, ProcessPoolExecutor(max_workers=args.workers,
            initializer=extractor.initialize,
            initargs=tuple(args.data_root / plan[k] for k in ['backbone', 'window', 'mel_filter'])) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row) + '\n')
            if number % 100 == 0 or number == len(jobs):
                log.flush()
                print(json.dumps({'completed': number, 'total': len(jobs), 'seconds': time.monotonic() - started}), flush=True)
    (args.output / 'complete.json').write_text(json.dumps({**identity, 'seconds': time.monotonic() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
