"""Audit all original training-only US synthesis floats and PCM on the Mac.

This does not synthesize speech, infer a wake model or access audio hardware.
"""
import argparse
import hashlib
from importlib.metadata import version
import io
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import wave

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def child(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, 'Unsafe artifact path')
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), 'Artifact escapes input directory')
    return resolved


def pcm16(source):
    with wave.open(source, 'rb') as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(),
                 wav.getcomptype()) == (1, 2, 16000, 'NONE'), 'Unexpected WAV geometry')
        frames = wav.getnframes()
        raw = wav.readframes(frames)
        require(len(raw) == frames * 2 and not wav.readframes(1), 'Incomplete PCM payload')
    return np.frombuffer(raw, dtype='<i2')


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'A fresh output report is required')
    plan_path = Path(__file__).with_name('plans') / 'us-coverage-mac-integrity-v1.json'
    plan = json.loads(plan_path.read_bytes())
    versions = {key: version(key) for key in plan['host_versions']}
    require(versions == plan['host_versions'], 'Use the preregistered package versions')
    paths = {'source': args.source / 'source.json', 'run': args.data / 'run.json',
             'manifest': args.data / 'manifest.jsonl', 'report': args.data / 'report.json'}
    for name, path in paths.items():
        require(sha(path) == plan[name + '_sha256'], 'Changed original metadata: ' + name)
    source = json.loads(paths['source'].read_bytes())
    run = json.loads(paths['run'].read_bytes())
    original = json.loads(paths['report'].read_bytes())
    require(run['versions'] == original['versions'], 'Original package reports disagree')
    require(all(run['versions'][key] == value for key, value in versions.items()),
            'Original and Mac package versions differ')
    for name in ('run', 'manifest', 'source'):
        require(original[name + '_sha256'] == plan[name + '_sha256'], 'Broken report link')
    require(run['source_sha256'] == plan['source_sha256']
            and run['script_sha256'] == original['script_sha256'], 'Broken generation identity')
    for name, info in source['files'].items():
        path = child(args.source, name)
        require(path.stat().st_size == info['size_bytes'] and sha(path) == info['sha256'],
                'Changed source artifact: ' + name)
    rows = [json.loads(line) for line in paths['manifest'].read_text().splitlines()]
    grid = plan['grid']
    expected = list(itertools.product(grid['voices'], grid['length_multipliers'],
                                      grid['target_texts'], grid['draws']))
    require(len(rows) == len(expected) == original['clips'] == 108, 'Incomplete original grid')
    require(len(list(args.data.glob('*.wav'))) == 108
            and len(list(args.data.glob('*.npy'))) == 216, 'Missing or extra waveform files')
    artifacts = set()
    # Authenticate all original bytes before decoding or resampling any source.
    for index, (row, entry) in enumerate(zip(rows, expected, strict=True)):
        require((row['voice'], row['length_multiplier'], row['target'], row['draw']) == entry
                and row['order'] == index, 'Original ordered grid differs')
        require(row['split'] == 'training' and row['role'] == 'us-coverage-training-only'
                and row['speaker_id'] == 0
                and row['text'] == grid['target_texts'][row['target']]
                and row['run_sha256'] == plan['run_sha256'], 'Unexpected source role or identity')
        for field, digest in (('native_path', 'native_sha256'),
                              ('float16k_path', 'float16k_sha256'), ('path', 'sha256')):
            path = child(args.data, row[field])
            require(path not in artifacts and sha(path) == row[digest], 'Changed/duplicate output')
            artifacts.add(path)
    details = []
    for row in rows:
        model_name = row['voice'] + '/en_US-' + row['voice'] + '-medium.onnx'
        model, config = child(args.source, model_name), child(args.source, model_name + '.json')
        require(sha(model) == row['model_sha256'] and sha(config) == row['config_sha256'],
                'Original model/config link differs')
        cfg = json.loads(config.read_bytes())
        ids = cfg['phoneme_id_map']
        require(all(phone in ids for phone in row['phonemes']), 'Unknown recorded phone')
        tokens = list(ids['^']) + list(ids['_'])
        for phone in row['phonemes']:
            tokens += list(ids[phone]) + list(ids['_'])
        require(tokens + list(ids['$']) == row['token_ids'], 'Recorded token construction differs')
        scales = np.asarray([cfg['inference']['noise_scale'],
                             cfg['inference']['length_scale'] * row['length_multiplier'],
                             cfg['inference']['noise_w']], dtype=np.float32)
        require(np.array_equal(scales.astype(np.float64), row['scales']), 'Original scales differ')
        native = np.load(child(args.data, row['native_path']), allow_pickle=False)
        saved = np.load(child(args.data, row['float16k_path']), allow_pickle=False)
        require(native.ndim == saved.ndim == 1 and native.dtype.kind == saved.dtype.kind == 'f'
                and len(native) and len(saved), 'Unexpected native/16k array geometry')
        rate = cfg['audio']['sample_rate']
        require(rate == row['native_rate'], 'Native sample rate differs')
        divisor = math.gcd(rate, 16000)
        replay = resample_poly(native, 16000 // divisor, rate // divisor)
        pcm = pcm16(str(child(args.data, row['path'])))
        require(hashlib.sha256(pcm.tobytes()).hexdigest() == row['pcm_sha256'], 'PCM hash differs')
        require(len(native) == row['native_samples']
                and len(saved) == len(pcm) == row['samples'] == row['generated_reference_end_sample']
                and len(saved) == math.ceil(len(native) * 16000 / rate), 'Original length differs')
        require(not row['trimmed'] and not row['normalized'] and row['native_durations'] is None,
                'Original processing declaration differs')
        buffer = io.BytesIO()
        sf.write(buffer, saved, 16000, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        serialized = pcm16(buffer)
        difference = replay.astype(np.float64) - saved.astype(np.float64)
        rms = float(np.sqrt(np.mean(saved.astype(np.float64) ** 2)))
        checks = dict(finite=bool(np.isfinite(native).all() and np.isfinite(saved).all()),
                      nonzero=bool(np.any(pcm)), duration=.2 <= len(pcm) / 16000 <= 5,
                      rms=rms >= .001, native_no_fullscale=bool(np.all(np.abs(native) < 1)),
                      resampled_no_fullscale=bool(np.all(np.abs(saved) < 1)),
                      pcm_no_endpoint=not bool(np.any((pcm == -32768) | (pcm == 32767))),
                      resample_exact=bool(np.array_equal(replay, saved)
                                          and replay.dtype == saved.dtype),
                      serialized_pcm_exact=bool(np.array_equal(serialized, pcm)))
        details.append(dict(path=row['path'],samples=len(pcm),native_dtype=str(native.dtype),
                            float16k_dtype=str(saved.dtype),checks=checks,
                            resample_changed_samples=int(np.count_nonzero(difference)),
                            resample_max_abs_difference=finite_number(np.max(np.abs(difference))),
                            pcm_changed_samples=int(np.count_nonzero(serialized != pcm)),
                            rms=finite_number(rms),original_rms=row['rms'],
                            all_checks_pass=all(checks.values())))
    result = dict(scope=plan['scope'],accepted=False,clips=len(details),
                  process_agreement_clips=sum(row['all_checks_pass'] for row in details),
                  all_process_checks_pass=all(row['all_checks_pass'] for row in details),
                  source_artifacts_authenticated=len(source['files']),output_artifacts_authenticated=len(artifacts),
                  input_sha256={name:sha(path) for name,path in paths.items()},
                  plan_sha256=sha(plan_path),script_sha256=sha(Path(__file__)),
                  host=dict(python=sys.version,platform=platform.platform(),versions=versions,
                            libsndfile=sf.__libsndfile_version__),details=details,
                  limits=[plan['limits'],
                          'PCM conversion repeats soundfile/libsndfile; not an independent rounding algorithm.',
                          'Token construction is checked from recorded phones; no independent phonemizer or pronunciation check.',
                          'RMS values are independently recomputed; their exact bit pattern is not a process-agreement gate.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:result[key] for key in ('clips','process_agreement_clips','all_process_checks_pass')}))


if __name__ == '__main__':
    main()
