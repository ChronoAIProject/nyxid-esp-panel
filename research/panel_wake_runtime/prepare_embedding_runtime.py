"""Prepare a private, hash-bound embedding diagnostic without editing app state."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    expected = {'sample_rate': 16000, 'step_samples': 1280, 'history_embeddings': 16,
                'minimum_end': 25, 'smoothing_steps': 3, 'refractory_steps': 25,
                'prime_samples': 38400, 'rearm_fraction': .5, 'score_denominator': 1536}
    if manifest['format'] != 'nyx-embedding-equal-mean-v1' or any(manifest[k] != v for k, v in expected.items()):
        raise ValueError('Unreviewed embedding runtime/event policy')
    if digest(args.calibration) != manifest['calibration_sha256'] or digest(args.model) != manifest['files']['dense.tflite']:
        raise ValueError('Explicit build model/calibration differ from bundle')
    calibration = json.loads(args.calibration.read_text())
    threshold = manifest['threshold']
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1 or threshold != calibration['selected']['threshold']:
        raise ValueError('Invalid or unfrozen threshold')
    temporal_identity = manifest['files']['temporal.tflite']
    layout = manifest.get('temporal_layout')
    if layout is not None:
        if (layout.get('output_sha256') != temporal_identity or layout.get('cases') != 299
                or layout.get('mismatches') != 0 or layout.get('p4_numeric_outputs') != 897
                or layout.get('p4_numeric_mismatches') != 0
                or layout.get('constant_buffer_bytes_unchanged') is not True
                or layout.get('accumulator_bound_safe') is not True
                or layout.get('ordered_products_equivalent') is not True):
            raise ValueError('Temporal layout requires exact host and P4 numerical verification')
        temporal_identity = layout['source_sha256']
    if [m['head_sha256'] for m in calibration['members']] != [manifest['files']['dense.tflite'], temporal_identity]:
        raise ValueError('Calibration member order/identity mismatch')
    backbone_layout = manifest.get('backbone_layout')
    if backbone_layout is not None:
        if (backbone_layout.get('output_sha256') != manifest['files']['backbone.tflite']
                or backbone_layout.get('cases') != 114
                or backbone_layout.get('source_mismatches') != 0
                or backbone_layout.get('rewritten_mismatches') != 0
                or backbone_layout.get('existing_constant_buffers_unchanged') is not True
                or backbone_layout.get('accumulator_bound_safe') is not True
                or backbone_layout.get('sample_rate_or_cadence_changed') is not False
                or backbone_layout.get('p4_verified') is not True
                or backbone_layout.get('p4_outputs') != 32832
                or backbone_layout.get('p4_mismatches') != 0):
            raise ValueError('Backbone padding rewrite requires exact host and P4 numerical verification')
    names = {'backbone.tflite', 'dense.tflite', 'temporal.tflite', 'native-coefficients.h'}
    if set(manifest['files']) != names:
        raise ValueError('Unexpected bundle file list')
    for name in names:
        path = args.bundle / name
        if digest(path) != manifest['files'][name] or path.stat().st_size != manifest['bytes'][name]:
            raise ValueError('Bundle file identity differs')
    args.output.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name in ('backbone', 'dense', 'temporal'):
        raw = (args.bundle / (name + '.tflite')).read_bytes()
        if not 1024 <= len(raw) <= 512 * 1024 or raw[4:8] != b'TFL3':
            raise ValueError('Expected bounded TFLite artifact')
        packed = struct.pack('<I', len(raw)) + zlib.compress(raw, 9)
        if zlib.decompress(packed[4:]) != raw:
            raise ValueError('Compression round trip differs')
        (args.output / f'embedding_{name}.bin').write_bytes(packed)
        sizes[name] = len(packed)
    (args.output / 'native-coefficients.h').write_bytes((args.bundle / 'native-coefficients.h').read_bytes())
    (args.output / 'embedding_config.h').write_text(
        '#pragma once\n'
        f'constexpr double embedding_threshold = {threshold:.17g};\n'
        f'constexpr char embedding_bundle_sha256[] = "{digest(args.bundle / "manifest.json")}";\n')
    (args.output / 'embedding-build-manifest.json').write_text(json.dumps({
        'scope': 'Experimental whole-app build only; no recognition acceptance',
        'bundle_sha256': digest(args.bundle / 'manifest.json'),
        'calibration_sha256': digest(args.calibration), 'compressed_bytes': sizes,
        'source_files': manifest['files'], 'event_policy': expected,
        'threshold': threshold, 'temporal_layout': layout,
        'backbone_layout': backbone_layout, 'accepted': False,
    }, indent=2) + '\n')
    print(json.dumps({'compressed_model_bytes': sizes, 'threshold': threshold}))


if __name__ == '__main__':
    main()
