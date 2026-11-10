"""Add generic five-band channel-EQ views to completed training-only mixes.

The original environmental renderer, room/noise RNG draws and original views
stay unchanged. A separate RNG supplies generic EQ gains, never a fitted room
response. No microphone recording is loaded and no sample rate/speed changes.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import sosfilt

import prepare_environment_views as environment

ORIGINAL_EXTRACT = environment.extractor.extract
EQ_STATE = None
POLICY = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equalize(pcm, gains, policy):
    """RBJ peaking biquads, global RMS restoration and explicit peak limiting."""
    coefficients = []
    for hz, gain in zip(policy['centers_hz'], gains):
        amplitude = 10 ** (gain / 40)
        omega = 2 * np.pi * hz / 16000
        alpha = np.sin(omega) / (2 * policy['q'])
        a0 = 1 + alpha / amplitude
        coefficients.append(np.asarray([1 + alpha * amplitude, -2 * np.cos(omega),
            1 - alpha * amplitude, a0, -2 * np.cos(omega), 1 - alpha / amplitude]) / a0)
    original = np.asarray(pcm, np.float64)
    filtered = sosfilt(np.asarray(coefficients), original)
    before = float(np.sqrt(np.mean(original ** 2)))
    after = float(np.sqrt(np.mean(filtered ** 2)))
    if min(before, after) <= 0 or not np.isfinite(filtered).all():
        raise ValueError('Silent or invalid channel view')
    restore = before / after
    filtered *= restore
    peak = float(np.abs(filtered).max())
    limit = min(1., policy['peak_ceiling'] / peak)
    result = np.rint(filtered * limit).astype(np.int16)
    if result.shape != pcm.shape or np.abs(result.astype(np.int32)).max() > policy['peak_ceiling']:
        raise ValueError('EQ changes duration or exceeds explicit peak ceiling')
    return result, {'original_rms': before, 'rms_restore_scale': restore,
                    'peak_limit_scale': limit, 'output_rms': float(np.sqrt(np.mean(result.astype(np.float64) ** 2)))}


def extract_equalized(pcm):
    if EQ_STATE['audit_reference']:
        reference = Path(EQ_STATE['reference_features'])
        if digest(reference) != EQ_STATE['reference_features_sha256']:
            raise ValueError('Completed reference array identity differs')
        if not np.array_equal(ORIGINAL_EXTRACT(pcm), np.load(reference, allow_pickle=False)):
            raise ValueError('Unmodified environmental render differs from completed reference')
    equalized, metadata = equalize(pcm, EQ_STATE['gains_db'], POLICY)
    metadata.update(equalized_pcm_sha256=hashlib.sha256(equalized.tobytes()).hexdigest(),
                    equalized_samples=len(equalized),
                    equalized_peak=int(np.abs(equalized.astype(np.int32)).max()))
    EQ_STATE['metadata'] = metadata
    return ORIGINAL_EXTRACT(equalized)


def initialize(model, window, mel_filter, backgrounds, policy, channel_policy):
    global POLICY
    environment.initialize(model, window, mel_filter, backgrounds, policy)
    POLICY = channel_policy
    # This process runs one job at a time. Only its opt-in extraction hook
    # changes; the imported reference renderer and RNG sequence stay exact.
    environment.extractor.extract = extract_equalized


def worker(job):
    global EQ_STATE
    rng = np.random.default_rng(int(job['id'][:16], 16))
    EQ_STATE = {key: job[key] for key in ('audit_reference', 'reference_features', 'reference_features_sha256')}
    EQ_STATE['gains_db'] = rng.uniform(*POLICY['gain_db'], len(POLICY['centers_hz'])).tolist()
    row = environment.worker({**job, 'id': job['reference_id']})
    for field in ('background_sha256', 'background_start'):
        if row[field] != job['reference_' + field]:
            raise ValueError('Original paired background draw differs')
    # The reference renderer's hash identifies the pre-EQ environmental mix.
    # Name it explicitly; the new model input has its own hash above.
    row['original_mix_pcm_sha256'] = row.pop('mixed_pcm_sha256')
    row.update(id=job['id'], reference_id=job['reference_id'], eq_gains_db=EQ_STATE['gains_db'],
               reference_features_exact=True if job['audit_reference'] else None, **EQ_STATE['metadata'])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--limit', type=int, help='Deterministic extraction pilot only')
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error('Positive workers and pilot limit required')
    plan = json.loads(args.plan.read_text())
    channel = plan['channel_policy']
    if (channel['centers_hz'] != [125, 315, 800, 2000, 5000] or channel['gain_db'] != [-12, 12]
            or channel['q'] != 2 ** -.5 or channel['peak_ceiling'] != 30000
            or plan['reference_audit_prefix'] != 48):
        raise ValueError('Expected the prespecified generic five-band policy')
    if digest(Path(__file__)) != plan['extractor_sha256']:
        raise ValueError('EQ extractor differs from frozen plan')
    for name, expected in plan['input_files'].items():
        if digest(args.data_root / name) != expected:
            raise ValueError('Frozen input identity differs')
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Reference implementation differs')
    original = json.loads((args.data_root / plan['reference_plan']).read_text())
    if any(plan['input_files'].get(k) != v for k, v in original['input_files'].items()):
        raise ValueError('Original extraction dependencies are not fully bound')
    tts = json.loads((args.data_root / 'head-features-tts-v1/jobs-private.json').read_text())
    negatives = json.loads((args.data_root / 'head-features-negatives-v1/jobs-private.json').read_text())
    sources = [r for r in tts if r['split'] == 'training' and r['variant'] == 0]
    backgrounds = [r for r in negatives if r['split'] == 'training' and r['kind'] == 'environment']
    heldout = {r['source_sha256'] for r in tts + negatives if r['split'] != 'training'}
    if not sources or not backgrounds or heldout.intersection(r['source_sha256'] for r in sources + backgrounds):
        raise ValueError('Empty training pool or held-out source overlap')
    references = {r['id']: r for r in map(json.loads, (args.data_root / plan['reference_index']).read_text().splitlines())}
    args.output = args.output.resolve()
    original_hash, plan_hash = digest(args.data_root / plan['reference_plan']), digest(args.plan)
    jobs = []
    for source in sources:
        for variant in original['variants']:
            reference_id = hashlib.sha256(f"{source['id']}:{variant}:{original_hash}".encode()).hexdigest()
            previous = references[reference_id]
            if previous['split'] != 'training' or previous['source_sha256'] != source['source_sha256'] or previous['label'] != source['label']:
                raise ValueError('Paired source or label differs')
            identity = hashlib.sha256(f'{reference_id}:eq:{plan_hash}'.encode()).hexdigest()
            path = Path(previous['path'])
            jobs.append({**source, 'id': identity, 'variant': variant, 'reference_id': reference_id,
                'reference_background_sha256': previous['background_sha256'],
                'reference_background_start': previous['background_start'],
                'reference_features': str(path if path.is_absolute() else args.data_root / path),
                'reference_features_sha256': previous['features_sha256'],
                'output': str(args.output / 'training/tts' / (identity + '.npy'))})
    # A pilot must cover engines, labels and both room variants rather than
    # being only the first Piper positives in the original source index.
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['engine'], job['label'], job['variant'])].append(job)
    jobs = [job for batch in itertools.zip_longest(*(groups[key] for key in sorted(groups)))
            for job in batch if job is not None]
    for index, job in enumerate(jobs):
        job['audit_reference'] = index < plan['reference_audit_prefix']
    total = len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    if len({r['id'] for r in jobs}) != len(jobs):
        raise ValueError('Duplicate EQ view identity')
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {'scope': __doc__, 'all_training_only': True, 'model_sha256': original['model_sha256'],
        'plan_sha256': plan_hash, 'source_sha256': digest(Path(__file__)), 'jobs': len(jobs),
        'full_jobs': total, 'pilot_only': len(jobs) != total, 'audio_speed_changed': False,
        'channel_policy': plan['channel_policy'], 'reference_audit_prefix': min(plan['reference_audit_prefix'], len(jobs))}
    (args.output / 'identity.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'jobs-private.json').write_text(json.dumps(jobs) + '\n')
    paths = [args.data_root / original[k] for k in ('model', 'window', 'mel_filter')]
    started = time.monotonic()
    with (args.output / 'index.jsonl').open('x') as log, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize,
            initargs=(*paths, backgrounds, original['policy'], plan['channel_policy'])) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row) + '\n')
            if number % 100 == 0 or number == len(jobs):
                log.flush()
                print(json.dumps({'completed': number, 'total': len(jobs), 'seconds': time.monotonic() - started}), flush=True)
    (args.output / 'complete.json').write_text(json.dumps(
        {**metadata, 'seconds': time.monotonic() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
