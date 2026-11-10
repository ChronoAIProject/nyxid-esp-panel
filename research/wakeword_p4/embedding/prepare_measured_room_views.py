"""Compare matched synthetic/measured room training views at unchanged speed.

Both arms add one paired variant3 view per original training TTS source. The
control reuses its completed synthetic-room array. Treatment changes only the
room impulse response, preserving original source/noise/RNG draws, bandpass,
level, SNR, leading/tail silence and frontend. No microphone/test audio is read.
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
import soundfile as sf

import prepare_environment_views as environment

ORIGINAL_ROOM_RESPONSE = environment.room_response
ROOM = None
RIRS = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align_energy(pcm, energy):
    values = np.array(pcm, np.float64, copy=True)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or not np.isfinite(energy) or energy <= 0):
        raise ValueError('Invalid measured response or reference energy')
    peak = float(np.abs(values).max())
    if peak <= 0:
        raise ValueError('Silent measured response')
    onset = int(np.flatnonzero(np.abs(values) >= peak * .1)[0])
    values = values[onset:]
    values *= np.sqrt(energy / np.sum(values ** 2))
    return values.astype(np.float32), onset


def matched_room_response(rng):
    # Consume exactly the original room draws before replacing the response.
    original = ORIGINAL_ROOM_RESPONSE(rng)
    if ROOM is None:
        return original
    energy = float(np.sum(original.astype(np.float64) ** 2))
    response, onset = align_energy(ROOM['pcm'], energy)
    ROOM['response_metadata'] = {'rir_sha256': ROOM['sha256'], 'rir_onset_samples': onset,
        'rir_retained_samples': len(response), 'control_response_energy': energy,
        'treatment_response_energy': float(np.sum(response.astype(np.float64) ** 2))}
    return response


def initialize(model, window, mel_filter, backgrounds, policy, rir_root, rir_records):
    global RIRS
    environment.initialize(model, window, mel_filter, backgrounds, policy)
    RIRS = []
    for row in rir_records:
        path = (Path(rir_root) / row['path']).resolve()
        path.relative_to(Path(rir_root).resolve())
        if row['split'] != 'training' or digest(path) != row['sha256']:
            raise ValueError('RIR identity or training role differs')
        pcm, rate = sf.read(path, dtype='float32')
        if rate != 16000 or pcm.ndim != 1 or len(pcm) != row['samples']:
            raise ValueError('Expected the exact mono16kHz measured response')
        align_energy(pcm, 1.)
        RIRS.append({**row, 'pcm': pcm})
    environment.room_response = matched_room_response


def worker(job):
    global ROOM
    if job['audit_reference']:
        ROOM = None
        control = environment.worker({**job, 'id': job['reference_id'],
                                      'output': job['control_audit_output']})
        if control['features_sha256'] != job['reference_features_sha256']:
            raise ValueError('Unmodified paired environmental replay differs')
    choice = np.random.default_rng(int(job['id'][:16], 16)).integers(len(RIRS))
    ROOM = dict(RIRS[int(choice)])
    row = environment.worker({**job, 'id': job['reference_id']})
    for name in ('background_sha256', 'background_start'):
        if row[name] != job['reference_' + name]:
            raise ValueError('Paired original background draw differs')
    row.update(id=job['id'], reference_id=job['reference_id'],
               reference_features_exact=True if job['audit_reference'] else None,
               **ROOM['response_metadata'])
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--control-output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--limit', type=int, help='Extraction pilot only, never fitting')
    args = p.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        p.error('Positive worker/pilot count required')
    plan = json.loads(args.plan.read_text())
    if digest(Path(__file__)) != plan['extractor_sha256']:
        raise ValueError('Implementation differs from the frozen plan')
    for name, expected in plan['helpers'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Frozen reference helper differs')
    base = args.data_root.resolve()
    for name, expected in plan['input_files'].items():
        if digest(base / name) != expected:
            raise ValueError('Frozen source/model/RIR metadata differs')
    original = json.loads((base / plan['reference_plan']).read_text())
    rirs = json.loads((base / plan['rir_audit']).read_text())
    if (not rirs['all_pinned_hashes_verified'] or len(rirs['records']) != 270
            or rirs['revision'] != plan['rir_revision'] or plan['reference_audit_prefix'] != 48):
        raise ValueError('Unexpected measured-response population or audit policy')
    tts = json.loads((base / 'head-features-tts-v1/jobs-private.json').read_text())
    negatives = json.loads((base / 'head-features-negatives-v1/jobs-private.json').read_text())
    sources = [r for r in tts if r['split'] == 'training' and r['variant'] == 0]
    backgrounds = [r for r in negatives if r['split'] == 'training' and r['kind'] == 'environment']
    heldout = {r['source_sha256'] for r in tts + negatives if r['split'] != 'training'}
    if not sources or not backgrounds or heldout.intersection(r['source_sha256'] for r in sources + backgrounds):
        raise ValueError('Empty training pool or held-out source overlap')
    reference = {r['id']: r for r in map(json.loads, (base / plan['reference_index']).read_text().splitlines())}
    plan_hash, original_hash = digest(args.plan), digest(base / plan['reference_plan'])
    output, control_output = args.output.resolve(), args.control_output.resolve()
    if output == control_output or output.exists() or control_output.exists():
        p.error('Separate fresh control/treatment outputs required')
    jobs, controls = [], {}
    for source in sources:
        ref_id = hashlib.sha256(f"{source['id']}:3:{original_hash}".encode()).hexdigest()
        ref = reference[ref_id]
        if (ref['split'] != 'training' or ref['source_sha256'] != source['source_sha256']
                or ref['label'] != source['label'] or ref['variant'] != 3):
            raise ValueError('Paired reference source/label differs')
        ref_path = Path(ref['path'])
        ref_path = ref_path if ref_path.is_absolute() else base / ref_path
        if digest(ref_path) != ref['features_sha256']:
            raise ValueError('Original paired control array differs')
        identity = hashlib.sha256(f'{ref_id}:measured-room:{plan_hash}'.encode()).hexdigest()
        controls[identity] = {**ref, 'id': identity, 'reference_id': ref_id,
                              'path': str(ref_path), 'paired_arm': 'synthetic_control'}
        jobs.append({**source, 'id': identity, 'variant': 3, 'reference_id': ref_id,
            'reference_features_sha256': ref['features_sha256'],
            'reference_background_sha256': ref['background_sha256'],
            'reference_background_start': ref['background_start'],
            'output': str(output / 'training/tts' / (identity + '.npy')),
            'control_audit_output': str(output / 'audit-controls' / (identity + '.npy'))})
    groups = defaultdict(list)
    for job in jobs:
        groups[(job['engine'], job['label'])].append(job)
    jobs = [r for batch in itertools.zip_longest(*(groups[k] for k in sorted(groups)))
            for r in batch if r is not None]
    total = len(jobs)
    if args.limit:
        jobs = jobs[:args.limit]
    for index, job in enumerate(jobs):
        job['audit_reference'] = index < plan['reference_audit_prefix']
    metadata = {'scope': __doc__, 'all_training_only': True, 'model_sha256': original['model_sha256'],
        'plan_sha256': plan_hash, 'source_sha256': digest(Path(__file__)), 'jobs': len(jobs),
        'full_jobs': total, 'pilot_only': len(jobs) != total, 'audio_speed_changed': False,
        'reference_audit_prefix': min(plan['reference_audit_prefix'], len(jobs)),
        'rir_revision': rirs['revision'], 'rir_audit_sha256': digest(base / plan['rir_audit'])}
    for folder, arm in ((control_output, 'synthetic_control'), (output, 'measured_treatment')):
        folder.mkdir(parents=True)
        (folder / 'identity.json').write_text(json.dumps({**metadata, 'paired_arm': arm}, indent=2) + '\n')
        (folder / 'jobs-private.json').write_text(json.dumps(jobs) + '\n')
    paths = [base / original[k] for k in ('model', 'window', 'mel_filter')]
    started = time.monotonic()
    with (output / 'index.jsonl').open('x') as log, (control_output / 'index.jsonl').open('x') as control_log, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize,
            initargs=(*paths, backgrounds, original['policy'], base / plan['rir_root'], rirs['records'])) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row) + '\n')
            control_log.write(json.dumps(controls[row['id']]) + '\n')
            if number % 100 == 0 or number == len(jobs):
                log.flush(); control_log.flush()
                print(json.dumps({'completed': number, 'total': len(jobs), 'seconds': time.monotonic() - started}), flush=True)
    for folder, arm in ((control_output, 'synthetic_control'), (output, 'measured_treatment')):
        (folder / 'complete.json').write_text(json.dumps({**metadata, 'paired_arm': arm,
            'seconds': time.monotonic() - started}, indent=2) + '\n')


if __name__ == '__main__':
    main()
