"""Package only the original optional C161/B40/history research handoff.

This opt-in build is separate from the shipping C161 build. Hashes identify the
three frozen exports; they are not a recognition or numerical acceptance gate.
No training, calibration, PCM or expected score generation occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path


MODELS = {
    'c161': (60952, 'c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'),
    'binary40': (64800, '454058039dd41e729ce82ac7f33a84edec4af9554615a632e3b33d47de38009c'),
    'history_expert': (64800, 'd92b95278334043ff70c20c6e8f267e13fe83acac7dbcf705cc2177df9130950'),
}


def prepare(handoff, output):
    files = {}
    for name, (size, expected) in MODELS.items():
        model = (Path(handoff) / 'models' / (name + '.tflite')).read_bytes()
        if len(model) != size or hashlib.sha256(model).hexdigest() != expected:
            raise ValueError(f'Expected the original {name} model; research variants must not replace it')
        files[name + '.bin'] = model
    manifest = {
        'schema': 'panel-c161-parallel-research-v1',
        'models': {name: {'bytes': size, 'sha256': sha} for name, (size, sha) in MODELS.items()},
        'sample_rate_hz': 16000, 'prime_samples': 81600,
        'threshold': 235 / 256, 'smoothing_steps': 5, 'refractory_steps': 67,
        'priority': list(MODELS),
        'scope': 'Opt-in research runtime; P4 resources, recognition and first-command acceptance pending',
    }
    files['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        if ({p.name for p in output.iterdir()} != set(files) or
                any((output / name).read_bytes() != data for name, data in files.items())):
            raise FileExistsError('Use a clean parallel build directory')
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (output / name).write_bytes(data)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.handoff, args.output), indent=2))
