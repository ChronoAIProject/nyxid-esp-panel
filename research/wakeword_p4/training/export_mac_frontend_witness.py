"""Export a hash-bound Mac frontend/input witness without audio I/O.

This reproduces one already-consumed regression source. It is a diagnostic
handoff, not a new benchmark, calibration run or model acceptance test.
"""
import argparse
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path
import platform
import sys
import zlib

import numpy as np
import soundfile as sf

from evaluate import Detector
from features import features


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'model-dir', 'reference', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source', default='rendered/Tessa-1-3.wav')
    args = parser.parse_args()
    reference = args.reference
    # Bind the unchanged, publicly archived Mac run rather than mutable reports.
    assert sha((reference/'result.json').read_bytes()) == (
        '47ccac0cfd8d9b3809aa62e6b9f8232e70ea1d12655959648a4067b3a608c57b')
    assert sha((reference/'scores.npz').read_bytes()) == (
        '374a399faa3f2acfc0603ef358767d08844f811f1d59f595c7fb7c90dacd8f29')
    result = json.loads((reference/'result.json').read_bytes())
    plan = json.loads((reference/'plan.json').read_bytes())
    assert sha((reference/'plan.json').read_bytes()) == result['plan_sha256']
    for name, expected in plan['host_versions'].items():
        assert version(name) == expected, name
    assert sha((args.sources/'manifest.jsonl').read_bytes()) == result['manifest_sha256']
    assert sha((args.sources/'baseline.json').read_bytes()) == result['baseline_sha256']
    rows = [json.loads(line) for line in (args.sources/'manifest.jsonl').read_text().splitlines()]
    matches = [i for i, row in enumerate(rows) if row['path'] == args.source]
    assert len(matches) == 1
    index = matches[0]
    row, expected = rows[index], result['source_details'][index]
    assert all(expected[key] == value for key, value in row.items())
    relative = Path(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    source = args.sources/relative
    assert sha(source.read_bytes()) == row['sha256']
    pcm, rate = sf.read(source, dtype='int16')
    assert rate == plan['sample_rate'] and pcm.ndim == 1 and len(pcm) == row['samples']
    baseline = json.loads((args.sources/'baseline.json').read_bytes())['clips'][index]
    assert all(baseline[key] == value for key, value in row.items())
    crc = zlib.crc32(pcm.astype('<i2').tobytes())
    assert crc == baseline['result']['crc32']
    tail = (-len(pcm)) % plan['v30_block_samples']
    padded = np.pad(pcm, (plan['v30_prime_samples'], tail)).astype('<i2')
    raw = features(padded)
    assert list(raw.shape) == expected['frontend_shape']
    assert str(raw.dtype) == expected['frontend_dtype']
    assert sha(raw.tobytes()) == expected['frontend_sha256']
    np.testing.assert_array_equal(raw, features(padded))
    model = args.model_dir/'hey_nyx.tflite'
    assert sha(model.read_bytes()) == result['model_sha256']
    detector = Detector(model)
    scores = np.rint(detector.predict(raw)*256).astype(np.uint8)
    with np.load(reference/'scores.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(scores, saved[f'output_{index}'])
    assert sha(scores.tobytes()) == expected['quantized_output_sha256']
    # Export the exact input quantization used by Detector.predict, including
    # the last incomplete step separately from the actually invoked triplets.
    values = np.clip(raw.astype(np.float32)/25.6, 0, 26)
    inputs = np.clip(np.rint(values/detector.scale)+detector.zero, -128, 127).astype(np.int8)
    invoked = inputs[:len(scores)*3].reshape(-1, 1, 3, 40)
    # Bypass frontend and Detector.predict for an independent invocation pass.
    detector.reset()
    direct = []
    for triplet in invoked:
        detector.interpreter.set_tensor(detector.input['index'], triplet)
        detector.interpreter.invoke()
        direct.append(detector.interpreter.get_tensor(detector.output['index'])[0, 0])
    np.testing.assert_array_equal(scores, np.asarray(direct, dtype=np.uint8))
    arrays = {'frontend_uint16': raw.astype('<u2'), 'quantized_input_int8': inputs,
              'invoked_input_int8': invoked, 'output_uint8': scores}
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out/'arrays.npz', **arrays)
    with np.load(out/'arrays.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        for key, value in arrays.items():
            np.testing.assert_array_equal(saved[key], value)
    package = distribution('pymicro-features')
    binary = next(path for path in package.files if str(path).endswith('.so'))
    report = {
        'scope': 'Single consumed-source Mac native frontend/input witness; no acceptance claim.',
        'source': row, 'source_index': index, 'pcm_crc32': crc,
        'prime_samples': plan['v30_prime_samples'], 'tail_zero_samples': tail,
        'reference_result_sha256': sha((reference/'result.json').read_bytes()),
        'reference_scores_sha256': sha((reference/'scores.npz').read_bytes()),
        'plan_sha256': result['plan_sha256'], 'model_sha256': result['model_sha256'],
        'features_source_sha256': sha(Path(features.__code__.co_filename).read_bytes()),
        'detector_source_sha256': sha(Path(sys.modules[Detector.__module__].__file__).read_bytes()),
        'export_source_sha256': sha(Path(__file__).read_bytes()),
        'platform': platform.platform(), 'python': sys.version,
        'versions': {name: version(name) for name in plan['host_versions']},
        'frontend_wheel': package.read_text('WHEEL'),
        'frontend_binary_name': str(binary),
        'frontend_binary_sha256': sha(package.locate_file(binary).read_bytes()),
        'input_quantization': {'scale': detector.scale, 'zero': detector.zero},
        'output_quantization': {'scale': detector.output_scale, 'zero': detector.output_zero},
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'bytes': value.nbytes, 'sha256': sha(value.tobytes())}
                   for key, value in arrays.items()},
        'arrays_npz_sha256': sha((out/'arrays.npz').read_bytes()),
        'frontend_and_scores_match_original_mac_run': True,
        'fresh_frontend_reset_exact': True, 'direct_tensor_invocation_exact': True,
        'audio_io_board_and_configuration_changes': False,
    }
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
