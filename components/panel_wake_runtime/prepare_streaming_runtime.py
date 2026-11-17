"""Package a streaming wake model for development; judge recognition in use.

Only file identity and bounded runtime settings are checked here. The board
checks the model's tensor ABI, allocates its state and runs inference. This
path needs no training report, calibration decision or expected score trace.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'panel-streaming-wake-profile-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(profile_path, model_path, output):
    profile_bytes = Path(profile_path).read_bytes()
    profile = json.loads(profile_bytes)
    model = Path(model_path).read_bytes()
    model_sha = sha(model)
    if not 1024 <= len(model) <= 512 * 1024 or model[4:8] != b'TFL3':
        raise ValueError('Expected a bounded TFLite model')
    if profile['schema'] != SCHEMA or profile['model_sha256'] != model_sha:
        raise ValueError('Profile must identify this streaming model')
    frames, threshold, smoothing = (profile[key] for key in
                                    ('context_frames', 'threshold', 'smoothing_steps'))
    if type(frames) is not int or not 1 <= frames <= 1000:
        raise ValueError('Context must be 1..1000 frames for bounded startup priming')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must be finite and between zero and one')
    if type(smoothing) is not int or smoothing not in (3, 5, 7):
        raise ValueError('Unsupported smoothing window')
    requested_prime = (frames + 200) * 160
    # The existing wake owner primes in complete 480-sample calls, without
    # resetting the frontend, pending features or detector at the live join.
    actual_prime = ((requested_prime + 479) // 480) * 480
    manifest = dict(schema=SCHEMA, model_sha256=model_sha, model_bytes=len(model),
                    profile_sha256=sha(profile_bytes), context_frames=frames,
                    threshold=threshold, smoothing_steps=smoothing, refractory_steps=67,
                    sample_rate=16000, frontend_hop_samples=160, model_stride_frames=3,
                    requested_prime_samples=requested_prime, actual_prime_samples=actual_prime,
                    event_arithmetic='uint8-sum-binary64-v1',
                    scope='Development packaging only; recognition and device behavior require testing')
    config = ('#pragma once\n'
              f'#define PANEL_WAKE_MODEL_SHA256 "{model_sha}"\n'
              f'#define PANEL_WAKE_THRESHOLD {threshold:.17g}\n'
              f'#define PANEL_WAKE_SMOOTHING_STEPS {smoothing}\n'
              f'#define PANEL_WAKE_PRIME_SAMPLES {requested_prime}\n').encode()
    files = {'hey_nyx_model.bin': model, 'wake_config.h': config,
             'manifest.json': (json.dumps(manifest, indent=2) + '\n').encode()}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if ({p.name for p in output.iterdir()} != set(files) or
                any((output / name).read_bytes() != data for name, data in files.items())):
            raise FileExistsError('Use a clean build directory for this model/profile')
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('profile', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.profile, args.model, args.output), indent=2))
