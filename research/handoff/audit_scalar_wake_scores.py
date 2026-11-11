"""Reconstruct a matched scalar wake study from its saved development scores.

Uses runs whose immutable source snapshots contain evaluate_compact_sequence.py.
Checks cached accounting, not fresh inference. No fitting, audio or board access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(left, right):
    """Counts/structure exact; allow only float summation-order roundoff."""
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal(v, right[k]) for k, v in left.items()))
    if isinstance(left, list):
        return (isinstance(right, list) and len(left) == len(right)
                and all(equal(a, b) for a, b in zip(left, right, strict=True)))
    if isinstance(left, float):
        return isinstance(right, (int, float)) and np.isfinite(left) and abs(left-right) <= 1e-15
    return type(left) is type(right) and left == right


def reconstruct_clip(row, record):
    require(record['source'] == row['path']
            and all(record[k] == row[k] for k in ('engine', 'variant', 'label', 'text')),
            'Clip identity differs')
    count = row['frames']//3-4
    values = np.asarray(record['scores'])
    times = (3*(np.arange(count)+4)+5)*.01
    end = (row['keyword_frame']+2)*.01 if row['label'] else None
    require(values.shape == (count,) and np.isfinite(values).all()
            and np.all((values >= 0) & (values <= 1)), 'Invalid cached scores')
    require(np.array_equal(times, record['times']) and record['keyword_end'] == end,
            'Clip timebase differs')
    mask = ((times >= end-.1) & (times <= end+.8) if end is not None else np.ones(count, bool))
    maximum = float(values[mask].max()) if mask.any() else 0.
    require(maximum == record['max_score'], 'Clip maximum does not reproduce')
    return {'engine': row['engine'], 'source': row['path'], 'label': row['label'],
            'max_score': maximum}, count


def audit(args):
    root = args.artifacts.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    inputs = json.loads((root/'input-audit.json').read_bytes())
    require(inputs['plan_sha256'] == sha(root/'plan.json')
            and inputs['frozen_source_plan_and_launch_match'], 'Missing bound input audit')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    # Use the separately tested summary implementation, not evaluator-local
    # accounting. Record it because this read-only auditor is supplementary.
    source = root/'model-control/source'
    source_launch = json.loads((root/'model-control/launch.json').read_bytes())
    for name in ('evaluate_compact_sequence.py', 'compare_phone_distillation.py', 'evaluate.py'):
        require(sha(source/name) == source_launch['source_files_sha256'][name],
                'Cached-accounting source differs from launch: '+name)
    sys.path.insert(0, str(source))
    from evaluate_compact_sequence import summarize
    from compare_phone_distillation import operating_points
    from evaluate import events
    rows = json.loads((args.features/'features.json').read_bytes())
    selected = [r for r in rows if r['split'] == 'validation']
    reserved = {r['sha256'] for r in rows if r['split'] != 'validation'}
    ambient_rows = json.loads((args.features/'human.json').read_bytes())
    ambient_rows += json.loads((args.features/'ambient.json').read_bytes())
    expected_ambient = [(i, r) for i, r in enumerate(ambient_rows) if r['split'] == 'validation']
    comparison = json.loads((root/'comparison.json').read_bytes())
    fields = tuple(comparison['arms']['control']['grid'][0])
    arms, reports = {}, {}
    for arm in plan['arms']:
        output = root/f'validation-{arm}'
        cal = json.loads((output/'threshold.json').read_bytes())
        details = json.loads((output/'validation-details.json').read_bytes())
        model_sha = sha(root/f'model-{arm}/hey_nyx.tflite')
        require(sha(output/'threshold.json') == comparison['arms'][arm]['calibration_sha256']
                and sha(output/'validation-details.json') == comparison['arms'][arm]['details_sha256'],
                'Saved validation files differ from comparison')
        require(cal['model_sha256'] == comparison['arms'][arm]['model_sha256'] == model_sha
                and cal['smoothing_steps'] == 5 and cal['threshold_grid'] == plan['threshold_grid'],
                'Model/grid/policy identity differs')
        clips, clip_steps = [], 0
        for row, record in zip(selected, details['validation'], strict=True):
            require(row['sha256'] not in reserved, 'Validation source overlaps another split')
            clip, count = reconstruct_clip(row, record)
            clips.append(clip)
            clip_steps += count
        ambient, continuous_steps, arrays = [], 0, {}
        for (number, row), item in zip(expected_ambient, details['ambient'], strict=True):
            raw = np.load(args.features/row['path'], allow_pickle=False, mmap_mode='r')
            count = len(raw)//3-4
            name = f'ambient-validation-{number:03d}.npy'
            values = np.load(output/name, allow_pickle=False)
            source_type = row.get('source_type', 'speech' if 'speaker' in row else 'environment')
            require(values.shape == (count,) and np.isfinite(values).all()
                    and np.all((values >= 0) & (values <= 1)), 'Invalid continuous scores')
            measured = {'hours': len(raw)*.01/3600, 'source_type': source_type,
                        'events': {str(t): events(values, t) for t in plan['threshold_grid']}}
            require(measured == item, 'Continuous exposure/event counts do not reproduce')
            ambient.append(measured)
            arrays[name] = {'source': row['path'], 'sha256': sha(output/name), 'steps': count}
            continuous_steps += count
        grid = [summarize(clips, ambient, t) for t in plan['threshold_grid']]
        require(equal(grid, cal['grid']), 'Reconstructed full calibration grid differs: '+arm)
        arms[arm] = {'grid': [{k: point[k] for k in fields} for point in grid]}
        require(equal(arms[arm]['grid'], comparison['arms'][arm]['grid']),
                'Comparison grid differs: '+arm)
        reports[arm] = {'clip_count': len(clips), 'clip_steps': clip_steps,
            'continuous_sources': len(ambient), 'continuous_steps': continuous_steps,
            'calibration_sha256': sha(output/'threshold.json'),
            'details_sha256': sha(output/'validation-details.json'), 'continuous_arrays': arrays}
    gate = operating_points(arms, plan.get('control_fallback_reference'))
    require(all(equal(v, comparison[k]) for k, v in gate.items()), 'Advancement decision differs')
    return {'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root/'plan.json'),
        'input_audit_sha256': sha(root/'input-audit.json'),
        'comparison_sha256': sha(root/'comparison.json'),
        'summary_source_sha256': sha(source/'evaluate_compact_sequence.py'),
        'event_source_sha256': sha(source/'evaluate.py'),
        'gate_source_sha256': sha(source/'compare_phone_distillation.py'),
        'all_maxima_timebases_events_grids_and_gate_reproduced': True,
        'float_accounting_absolute_tolerance': 1e-15, 'arms': reports,
        'limits': 'Read-only cached-score reconstruction. Input audit separately binds the frozen plan/source/bank and optional full scalar cache. This does not replay INT8 invocations or establish independent wake, pronunciation, physical latency, P4 resource or call stability quality.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))


if __name__ == '__main__':
    main()
