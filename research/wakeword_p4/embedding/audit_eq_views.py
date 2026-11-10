"""Audit complete EQ views against immutable original and pilot feature arrays.

Does not score a detector or load microphone captures. Reports aggregate
provenance, split isolation, numeric bounds and deterministic pilot replay.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def array_path(root, row):
    path = Path(row['path'])
    return path if path.is_absolute() else root / path


def verify_array(root, row):
    path = array_path(root, row)
    if sha(path) != row['features_sha256']:
        raise ValueError('Feature checksum differs')
    array = np.load(path, allow_pickle=False)
    if array.shape != (row['frames'], 96) or array.dtype != np.int8 or len(array) < 25:
        raise ValueError('Invalid embedding array')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Fresh report output required')
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['input_files'].items():
        if sha(args.data_root / name) != expected:
            raise ValueError('Original input index/completion identity differs')
    if sha(Path(__file__).with_name('prepare_eq_views.py')) != plan['extractor_sha256']:
        raise ValueError('Extractor identity differs')
    for name, expected in plan['helpers'].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Original extractor dependency differs')
    complete = json.loads((args.features / 'complete.json').read_text())
    identity = json.loads((args.features / 'identity.json').read_text())
    for metadata in (complete, identity):
        if (metadata['pilot_only'] or metadata['plan_sha256'] != sha(args.plan)
                or not metadata['all_training_only'] or metadata['audio_speed_changed']):
            raise ValueError('Expected complete unchanged-speed training-only extraction')
    reference = {r['id']: r for r in rows(args.data_root / plan['reference_index'])}
    generated = rows(args.features / 'index.jsonl')
    if (len(generated) != complete['jobs'] or len(generated) != len(reference)
            or len({r['id'] for r in generated}) != len(generated)
            or len({r['reference_id'] for r in generated}) != len(reference)):
        raise ValueError('Reference view coverage or unique identity differs')
    original_rows = []
    for directory in ('head-features-tts-v1', 'head-features-negatives-v1'):
        original_rows.extend(rows(args.data_root / directory / 'index.jsonl'))
    heldout = {r['source_sha256'] for r in original_rows if r['split'] != 'training'}
    for row in [*original_rows, *reference.values()]:
        verify_array(args.data_root, row)
    audited, peak_limited, max_rms_error = 0, 0, 0.
    for row in generated:
        prior = reference[row['reference_id']]
        if (row['split'] != 'training' or row['source_sha256'] in heldout
                or row['background_sha256'] in heldout):
            raise ValueError('Held-out audio was used in training augmentation')
        for field in ('source_sha256', 'label', 'variant', 'engine', 'background_sha256',
                      'background_start', 'frames', 'seconds', 'source_seconds', 'keyword_end'):
            if row[field] != prior[field]:
                raise ValueError('Original source/noise/timebase pairing differs')
        if row['original_mix_pcm_sha256'] != prior['mixed_pcm_sha256']:
            raise ValueError('Original environmental mix PCM differs')
        gains = row['eq_gains_db']
        if len(gains) != 5 or not all(np.isfinite(g) and -12 <= g <= 12 for g in gains):
            raise ValueError('EQ gain exceeds frozen generic bounds')
        if (not 0 < row['equalized_peak'] <= 30000
                or row['equalized_samples'] != round(row['seconds'] * 16000)
                or row['equalized_pcm_sha256'] == row['original_mix_pcm_sha256']
                or not 0 < row['peak_limit_scale'] <= 1
                or row['original_rms'] <= 0 or row['output_rms'] <= 0
                or not np.isfinite(row['rms_restore_scale']) or row['rms_restore_scale'] <= 0):
            raise ValueError('EQ samples, amplitude accounting or PCM identity invalid')
        error = abs(row['output_rms'] / (row['original_rms'] * row['peak_limit_scale']) - 1)
        if not np.isfinite(error) or error > .002:
            raise ValueError('RMS preservation differs beyond PCM rounding tolerance')
        max_rms_error = max(max_rms_error, error)
        audited += row['reference_features_exact'] is True
        peak_limited += row['peak_limit_scale'] < 1
        verify_array(args.data_root, row)
    pilot = rows(args.pilot / 'index.jsonl')
    if len(pilot) != plan['reference_audit_prefix'] or audited != len(pilot):
        raise ValueError('Missing original-render array audit')
    for a, b in zip(pilot, generated):
        if {k: v for k, v in a.items() if k != 'path'} != {k: v for k, v in b.items() if k != 'path'}:
            raise ValueError('Pilot and full extraction differ')
    groups = Counter((r['engine'], r['label'], r['variant']) for r in pilot)
    if len(groups) != 24 or set(groups.values()) != {2}:
        raise ValueError('Pilot must cover six engines, both labels and variants equally')
    report = {'scope': __doc__, 'plan_sha256': sha(args.plan),
        'audit_script_sha256': sha(Path(__file__)), 'views': len(generated),
        'engines': dict(Counter(r['engine'] for r in generated)),
        'positive_views': sum(r['label'] for r in generated),
        'original_arrays_hash_verified': len(original_rows) + len(reference),
        'paired_source_background_timebase_pcm_exact': True,
        'pilot_and_full_rows_exact': len(pilot), 'original_render_features_exact': audited,
        'heldout_source_and_background_excluded': True,
        'peak_ceiling': 30000, 'peak_limited_views': peak_limited,
        'max_relative_rms_rounding_error': max_rms_error,
        'original_input_indexes_and_arrays_unchanged': True,
        'microphone_recordings_read': False, 'audio_speed_changed': False,
        'feature_index_sha256': sha(args.features / 'index.jsonl'),
        'complete_sha256': sha(args.features / 'complete.json'), 'passed': True,
        'limits': 'Extraction/provenance audit only. No recognition, P4 or human acceptance.'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
