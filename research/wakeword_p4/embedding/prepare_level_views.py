"""Add training-only level views paired with completed environmental views.

Reuse each original room, background, SNR and RNG draw. Only the planned speech
RMS range changes; original samples/rate/timebase and held-out rows stay intact.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import prepare_environment_views as environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(job):
    # The old ID is the original RNG seed; the new ID separates the new view.
    row = environment.worker({**job, 'id': job['reference_id']})
    for field in ('background_sha256', 'background_start'):
        if row[field] != job['reference_' + field]:
            raise ValueError('Paired background draw differs from original view')
    row.update(id=job['id'], reference_id=job['reference_id'])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='Deterministic extraction pilot only')
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error('Workers and an explicit pilot limit must be positive')
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['input_files'].items():
        if digest(args.data_root / name) != expected:
            raise ValueError('Frozen input identity differs: ' + name)
    if digest(Path(__file__)) != plan['extractor_sha256']:
        raise ValueError('Level extractor differs from frozen plan')
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Feature helper differs from frozen plan')
    original_path = args.data_root / plan['reference_plan']
    original = json.loads(original_path.read_text())
    policy = dict(original['policy'])
    level = plan['speech_rms_fraction']
    if len(level) != 2 or not 0 < level[0] < level[1] <= 1:
        raise ValueError('Expected an explicit increasing RMS fraction range')
    policy['speech_rms_fraction'] = level
    if digest(original_path) != plan['input_files'][plan['reference_plan']]:
        raise ValueError('Reference augmentation plan differs')
    # The level plan must bind every original extraction dependency as well.
    if any(plan['input_files'].get(k) != v for k, v in original['input_files'].items()):
        raise ValueError('Original extraction dependencies are not fully bound')
    tts = json.loads((args.data_root / 'head-features-tts-v1/jobs-private.json').read_text())
    negatives = json.loads((args.data_root / 'head-features-negatives-v1/jobs-private.json').read_text())
    heldout = {r['source_sha256'] for r in tts + negatives if r['split'] != 'training'}
    sources = [r for r in tts if r['split'] == 'training' and r['variant'] == 0]
    backgrounds = [r for r in negatives if r['split'] == 'training' and r['kind'] == 'environment']
    if not sources or not backgrounds or heldout.intersection(r['source_sha256'] for r in sources + backgrounds):
        raise ValueError('Empty training pool or held-out source overlap')
    index = args.data_root / plan['reference_index']
    if plan['reference_index'] not in plan['input_files']:
        raise ValueError('Reference index must be hash-bound')
    reference = {r['id']: r for r in map(json.loads, index.read_text().splitlines())}
    original_hash, plan_hash = digest(original_path), digest(args.plan)
    args.output = args.output.resolve()
    jobs = []
    for source in sources:
        for variant in original['variants']:
            previous_id = hashlib.sha256(f"{source['id']}:{variant}:{original_hash}".encode()).hexdigest()
            previous = reference[previous_id]
            if (previous['split'] != 'training' or previous['source_sha256'] != source['source_sha256']
                    or previous['label'] != source['label'] or previous['variant'] != variant):
                raise ValueError('Original paired view provenance differs')
            identity = hashlib.sha256(f'{previous_id}:level:{plan_hash}'.encode()).hexdigest()
            jobs.append({**source, 'id': identity, 'variant': variant, 'reference_id': previous_id,
                'reference_background_sha256': previous['background_sha256'],
                'reference_background_start': previous['background_start'],
                'output': str(args.output / 'training/tts' / (identity + '.npy'))})
    full_count = len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    if len({r['id'] for r in jobs}) != len(jobs):
        raise ValueError('Duplicate level-view identity')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'scope': __doc__, 'model_sha256': original['model_sha256'],
        'plan_sha256': plan_hash, 'source_sha256': digest(Path(__file__)),
        'all_training_only': True, 'jobs': len(jobs), 'full_jobs': full_count,
        'pilot_only': len(jobs) != full_count, 'speech_rms_fraction': level,
        'background_draws_verified_against_original': True, 'audio_speed_changed': False}
    (args.output / 'identity.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'jobs-private.json').write_text(json.dumps(jobs) + '\n')
    started = time.monotonic()
    paths = [args.data_root / original[k] for k in ('model', 'window', 'mel_filter')]
    with (args.output / 'index.jsonl').open('x') as log, ProcessPoolExecutor(
            max_workers=args.workers, initializer=environment.initialize,
            initargs=(*paths, backgrounds, policy)) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row) + '\n')
            if number % 100 == 0 or number == len(jobs):
                log.flush()
                print(json.dumps({'completed': number, 'total': len(jobs),
                    'seconds': time.monotonic() - started}), flush=True)
    (args.output / 'complete.json').write_text(json.dumps(
        {**metadata, 'seconds': time.monotonic() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
