"""Add training-only normal-speed views with verified environmental backgrounds.

Existing v1 views remain immutable. A frozen plan identifies the exact source
indexes, background jobs and augmentation policy before feature extraction.
Generated audio/features and source paths stay in an explicit private root.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import math
import time

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt
import soundfile as sf

import prepare_head_features as extractor

BACKGROUNDS = None
POLICY = None
VERIFIED = None


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def room_response(rng):
    """Second-order shoebox image paths, aligned to the direct arrival.

    This is a limited synthetic room response, not measured room acoustics.
    Preserve time/speed and the original keyword-end timebase.
    """
    dimensions = rng.uniform([3., 3., 2.4], [9., 8., 3.8])
    source = rng.uniform(.15, .85, 3)*dimensions
    microphone = rng.uniform(.15, .85, 3)*dimensions
    direct_distance = max(.3, float(np.linalg.norm(source-microphone)))
    reflection = rng.uniform(.3, .8)
    response = np.zeros(8000, np.float64)
    response[0] = 1
    for indices in itertools.product(range(-2, 3), repeat=3):
        order = sum(abs(i) for i in indices)
        if not 1 <= order <= 2:
            continue
        image = np.asarray([2*math.ceil(i/2)*length+(-1.)**i*at
                            for i, length, at in zip(indices, dimensions, source)])
        distance = float(np.linalg.norm(image-microphone))
        delay = max(0, round((distance-direct_distance)/343*16000))
        if delay < len(response):
            response[delay] += reflection**order*direct_distance/max(distance, .3)
    return response.astype(np.float32)


def initialize(model, window, mel_filter, backgrounds, policy):
    global BACKGROUNDS, POLICY, VERIFIED
    extractor.initialize(model, window, mel_filter)
    BACKGROUNDS, POLICY, VERIFIED = backgrounds, policy, set()


def verify(path, expected):
    key = (str(path), expected)
    if key not in VERIFIED:
        if digest(path) != expected:
            raise ValueError('Audio identity differs from the frozen input')
        VERIFIED.add(key)


def mix_view(pcm, noise, rng, variant, policy):
    """Return unchanged-length samples; never resample or change speaking speed."""
    signal = np.pad(pcm.astype(np.float64), (38400, 16000))
    if variant == 3:
        signal = fftconvolve(signal, room_response(rng))[:len(signal)]
        high = float(rng.uniform(*policy['highpass_hz']))
        low = float(rng.uniform(*policy['lowpass_hz']))
        signal = sosfilt(butter(2, [high, low], btype='bandpass', fs=16000, output='sos'), signal)
    speech_rms = max(655., float(np.sqrt(np.mean(pcm.astype(np.float64)**2))))
    level = float(rng.uniform(*policy['speech_rms_fraction']))*32767
    signal *= level/speech_rms
    noise = noise.astype(np.float64)
    noise -= noise.mean()
    noise_rms = float(np.sqrt(np.mean(noise**2)))
    if noise_rms < 1:
        raise ValueError('Selected background interval has insufficient signal')
    snr = float(rng.uniform(*policy['snr_db']))
    signal += noise/noise_rms*level*10**(-snr/20)
    return np.clip(np.rint(signal), -32768, 32767).astype(np.int16)


def worker(job):
    verify(job['source'], job['source_sha256'])
    pcm, rate = sf.read(job['source'], dtype='int16')
    if rate != 16000 or pcm.ndim != 1:
        raise ValueError('Expected original mono16kHz source')
    rng = np.random.default_rng(int(job['id'][:16], 16))
    length = len(pcm)+38400+16000
    # Pool is training environmental recordings only; draw a same-length slice.
    # Retry quiet intervals without silently switching to synthetic noise.
    for _ in range(20):
        background = BACKGROUNDS[int(rng.integers(len(BACKGROUNDS)))]
        verify(background['source'], background['source_sha256'])
        with sf.SoundFile(background['source']) as handle:
            if handle.samplerate != 16000 or handle.channels != 1 or len(handle) < length:
                raise ValueError('Expected sufficiently long mono16kHz background')
            start = int(rng.integers(len(handle)-length+1))
            handle.seek(start)
            noise = handle.read(length, dtype='int16')
        if float(np.std(noise.astype(np.float64))) >= 1:
            break
    else:
        raise ValueError('No usable interval found in the verified noise pool')
    mixed = mix_view(pcm, noise, rng, job['variant'], POLICY)
    embeddings = extractor.extract(mixed)
    output = Path(job['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)
    row = {key: job[key] for key in ['id', 'split', 'kind', 'label', 'source_sha256', 'variant', 'engine']}
    row.update({'path': str(output), 'frames': len(embeddings), 'seconds': len(mixed)/16000,
                'source_seconds': len(pcm)/16000,
                'keyword_end': (38400+job['keyword_end'])/16000 if job['label'] else None,
                'features_sha256': digest(output), 'mixed_pcm_sha256': hashlib.sha256(mixed.tobytes()).hexdigest(),
                'background_sha256': background['source_sha256'], 'background_start': start})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.workers < 1:
        parser.error('--workers must be positive')
    for name, expected in plan['input_files'].items():
        if digest(args.data_root/name) != expected:
            raise ValueError('Frozen input identity differs')
    if digest(Path(__file__)) != plan['extractor_sha256']:
        raise ValueError('Extractor differs from the frozen plan')
    if digest(Path(extractor.__file__)) != plan['base_extractor_sha256']:
        raise ValueError('Base extraction implementation differs')
    if digest(Path(extractor.__file__).with_name('native_mel.py')) != plan['frontend_sha256']:
        raise ValueError('Native frontend implementation differs')
    tts = json.loads((args.data_root/'head-features-tts-v1/jobs-private.json').read_text())
    negative = json.loads((args.data_root/'head-features-negatives-v1/jobs-private.json').read_text())
    heldout = {j['source_sha256'] for j in tts+negative if j['split'] != 'training'}
    backgrounds = [j for j in negative if j['split'] == 'training' and j['kind'] == 'environment']
    source = [j for j in tts if j['split'] == 'training' and j['variant'] == 0]
    if not backgrounds or not source:
        raise ValueError('Empty source or background pool')
    if heldout.intersection(j['source_sha256'] for j in source+backgrounds):
        raise ValueError('Held-out audio cannot be mixed into training')
    if set(plan['variants']) != {2, 3}:
        raise ValueError('Expected the prespecified environmental and room/EQ views')
    jobs = []
    plan_sha256 = digest(args.plan)
    for row in source:
        for variant in plan['variants']:
            job = dict(row)
            identity = hashlib.sha256(f"{row['id']}:{variant}:{plan_sha256}".encode()).hexdigest()
            job.update({'id': identity, 'variant': variant, 'output': str(args.output/'training/tts'/f'{identity}.npy')})
            jobs.append(job)
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate augmented identity')
    args.output.mkdir(exist_ok=False)
    (args.output/'jobs-private.json').write_text(json.dumps(jobs)+'\n')
    identity = {'scope': 'Training-only added environmental views; no held-out scoring',
                'model_sha256': plan['model_sha256'], 'plan_sha256': plan_sha256,
                'source_sha256': digest(Path(__file__)), 'jobs': len(jobs),
                'background_sources': len(backgrounds), 'all_training_only': True}
    (args.output/'identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    paths = [args.data_root/plan[key] for key in ['model', 'window', 'mel_filter']]
    if digest(paths[0]) != plan['model_sha256']:
        raise ValueError('Backbone differs from frozen identity')
    started = time.monotonic()
    with (args.output/'index.jsonl').open('x') as log, ProcessPoolExecutor(
            max_workers=args.workers, initializer=initialize,
            initargs=(*paths, backgrounds, plan['policy'])) as pool:
        for number, row in enumerate(pool.map(worker, jobs), 1):
            log.write(json.dumps(row)+'\n')
            if number % 100 == 0 or number == len(jobs):
                log.flush()
                print(json.dumps({'completed': number, 'total': len(jobs), 'seconds': time.monotonic()-started}), flush=True)
    complete = {**identity, 'seconds': time.monotonic()-started}
    (args.output/'complete.json').write_text(json.dumps(complete, indent=2)+'\n')
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
