"""Frozen-model gain/arrival-phase diagnostic on explicit captured PCM only.

This never selects a threshold, trains a model or changes live audio. Offset
adds silence before the identical ordered recording; gains keep sample rate,
sample count and speed. Reports are diagnostic, not independent wake trials.
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
from evaluate_equal_mean import EqualMeanHead
from evaluate_head import events

CONFIG = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialize(config):
    global CONFIG
    CONFIG = config
    extractor.initialize(config['backbone'], config['window'], config['mel_filter'])


def measure(job):
    pcm, rate = sf.read(CONFIG['pcm'], dtype='int16', always_2d=True)
    if rate != 16000:
        raise ValueError('Expected unchanged16kHz input')
    signal = pcm[:, CONFIG['channel']].astype(np.float64) * job['gain']
    clipped = int(np.count_nonzero((signal < -32768) | (signal > 32767)))
    signal = np.clip(np.rint(signal), -32768, 32767).astype(np.int16)
    signal = np.pad(signal, (38400 + job['offset_samples'], 0))
    started = time.monotonic()
    embeddings = extractor.extract(signal)
    model = EqualMeanHead([Path(name) for name in CONFIG['heads']])
    scores, times = model.scores(embeddings, 3)
    # Keep members separate to expose which frozen head limits the margin.
    member_peaks = [float(h.scores(embeddings, 3)[0].max()) for h in model.members]
    return {**job, 'clipped_samples': clipped, 'peak_score': float(scores.max()),
            'head_peak_scores': member_peaks,
            'events': events(scores, CONFIG['threshold']),
            'peak_time_s': float(times[scores.argmax()] - (38400 + job['offset_samples']) / 16000),
            'seconds': time.monotonic() - started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pcm', type=Path, required=True)
    p.add_argument('--channel', type=int, default=0)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--bundle', type=Path)
    selection.add_argument('--calibration', type=Path)
    p.add_argument('--data-root', type=Path, help='Explicit checkpoint root for a frozen equal-mean calibration')
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--window', type=Path, required=True)
    p.add_argument('--mel-filter', type=Path, required=True)
    p.add_argument('--gains', type=float, nargs='+', default=[1])
    p.add_argument('--offsets', type=int, nargs='+', default=[0, 160, 320, 640, 960, 1264])
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if (args.output.exists() or args.workers < 1 or any(not 0 < g <= 16 for g in args.gains)
            or any(not 0 <= o < 1280 for o in args.offsets)):
        p.error('Fresh output, positive workers, gains0..16 and offsets0..1279required')
    info = sf.info(args.pcm)
    if info.samplerate != 16000 or not 0 <= args.channel < info.channels:
        raise ValueError('Expected explicit valid channel in original16kHz recording')
    if args.bundle:
        identity = args.bundle / 'manifest.json'
        manifest = json.loads(identity.read_text())
        for name, expected in manifest['files'].items():
            if digest(args.bundle / name) != expected:
                raise ValueError('Runtime bundle identity differs')
        expected_backbone = manifest['backbone_layout']['source_sha256']
        heads = [args.bundle / 'dense.tflite', args.bundle / 'temporal.tflite']
        threshold = manifest['threshold']
    else:
        if args.data_root is None:
            p.error('--calibration requires --data-root')
        identity = args.calibration
        manifest = json.loads(identity.read_text())
        if len(manifest['members']) != 2:
            raise ValueError('Expected frozen equal mean of two members')
        heads = [args.data_root / member['checkpoint'] / 'head.tflite' for member in manifest['members']]
        for head, member in zip(heads, manifest['members']):
            if not head.resolve().is_relative_to(args.data_root.resolve()) or digest(head) != member['head_sha256']:
                raise ValueError('Frozen member identity differs')
        selected = manifest['selected']
        threshold = selected['threshold']
        expected_backbone = json.loads((heads[0].parent / 'validation/calibration.json').read_text())['backbone_sha256']
    if digest(args.backbone) != expected_backbone:
        raise ValueError('Expected the bundle source unfused reference backbone')
    config = {key: str(getattr(args, key)) for key in ('pcm', 'backbone', 'window', 'mel_filter')}
    config.update(channel=args.channel, threshold=threshold, heads=[str(head) for head in heads])
    jobs = [{'gain': gain, 'offset_samples': offset} for gain in args.gains for offset in args.offsets]
    report = {'scope': __doc__, 'input_sha256': digest(args.pcm), 'channel': args.channel,
              'configuration_sha256': digest(identity),
              'threshold': threshold, 'jobs_prespecified': jobs,
              'original_speed_and_sample_rate_unchanged': True, 'accepted': False, 'results': []}
    # Freeze jobs and identities before observing any diagnostic scores.
    plan = args.output.with_suffix('.plan.json')
    with plan.open('x') as stream:
        json.dump(report, stream, indent=2)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config,)) as pool:
        for row in pool.map(measure, jobs):
            report['results'].append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
