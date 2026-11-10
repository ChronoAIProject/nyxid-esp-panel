"""Verify complete matched training views before deriving any fitting plan."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Use a fresh audit output')
    base = args.data_root.resolve()
    plan = json.loads(args.plan.read_text())
    for name, expected in plan['input_files'].items():
        if digest(base / name) != expected:
            raise ValueError('Original source/index identity changed')
    dirs = ['head-features-measured-room-control-v1', 'head-features-measured-room-v1']
    complete = [json.loads((base / name / 'complete.json').read_text()) for name in dirs]
    if any(d['pilot_only'] or not d['all_training_only'] or d['jobs'] != 10070
           or d['plan_sha256'] != digest(args.plan) for d in complete):
        raise ValueError('Incomplete/nonmatching full paired extraction')
    rows = [list(map(json.loads, (base / name / 'index.jsonl').read_text().splitlines())) for name in dirs]
    if any(len(r) != 10070 or len({x['id'] for x in r}) != 10070 for r in rows):
        raise ValueError('Incomplete or duplicate paired coverage')
    reference = {r['id']: r for r in map(json.loads, (base / plan['reference_index']).read_text().splitlines())}
    jobs = json.loads((base / 'head-features-tts-v1/jobs-private.json').read_text())
    heldout = {r['source_sha256'] for r in jobs if r['split'] != 'training'}
    rirs = json.loads((base / plan['rir_audit']).read_text())
    allowed_rirs = {r['sha256'] for r in rirs['records'] if r['split'] == 'training'}
    changed, audit_count, error = 0, 0, 0.
    for a, b in zip(*rows):
        if any(a[k] != b[k] for k in ('id', 'reference_id', 'split', 'kind', 'label', 'source_sha256',
                'variant', 'engine', 'frames', 'seconds', 'source_seconds', 'keyword_end',
                'background_sha256', 'background_start')):
            raise ValueError('Source, label, timebase or background pairing differs')
        if a['split'] != 'training' or a['source_sha256'] in heldout or b['rir_sha256'] not in allowed_rirs:
            raise ValueError('Nontraining input in additional views')
        ref = reference[a['reference_id']]
        if ref['features_sha256'] != a['features_sha256']:
            raise ValueError('Control is not the exact original room feature')
        arrays = []
        for row in (a, b):
            path = Path(row['path'])
            path = path if path.is_absolute() else base / path
            if digest(path) != row['features_sha256']:
                raise ValueError('Feature payload hash differs')
            raw = np.load(path, allow_pickle=False)
            if raw.dtype != np.int8 or raw.shape != (row['frames'], 96):
                raise ValueError('Feature payload type/shape differs')
            arrays.append(raw)
        changed += int(not np.array_equal(*arrays))
        if not np.isclose(b['control_response_energy'], b['treatment_response_energy'], rtol=1e-6, atol=1e-8):
            raise ValueError('Response energy matching failed')
        error = max(error, abs(b['control_response_energy'] - b['treatment_response_energy']))
        if b['reference_features_exact'] is True:
            path = base / dirs[1] / 'audit-controls' / (b['id'] + '.npy')
            if digest(path) != a['features_sha256']:
                raise ValueError('Original numerical renderer replay differs')
            audit_count += 1
    if audit_count != plan['reference_audit_prefix']:
        raise ValueError('Missing prespecified original renderer replays')
    report = {'scope': __doc__, 'plan_sha256': digest(args.plan), 'pairs': len(rows[0]),
        'all_training_only': True, 'source_label_timebase_background_pairing_verified': True,
        'feature_payload_hashes_and_shapes_verified': True, 'original_control_arrays_exact': len(rows[0]),
        'original_control_numerical_replays_byte_exact': audit_count,
        'changed_treatment_feature_arrays': changed, 'maximum_response_energy_difference': error,
        'cohorts': [{'engine': e, 'label': label, 'sources': n} for (e, label), n in
                    sorted(Counter((r['engine'], r['label']) for r in rows[0]).items())],
        'feature_bindings': {name: {'index_sha256': digest(base / name / 'index.jsonl'),
                                   'complete_sha256': digest(base / name / 'complete.json')}
                             for name in dirs}, 'audio_speed_changed': False, 'accepted': False}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
