"""Read-only reconstruction of a frozen same-weight precision diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from audit_scalar_wake_scores import equal, reconstruct_clip, require, sha


def audit(args):
    root = args.artifacts.resolve()
    output = root/'evaluation'
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'launch.json').read_bytes())
    committed = subprocess.check_output(['git', 'show', launch['checkout_revision']+
        ':research/wakeword_p4/training/plans/scalar-precision-diagnostic-v1.json'], cwd=args.repository)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256']
            and sha(output/'plan.json') == sha(root/'plan.json'), 'Frozen plan/launch differs')
    command = launch['command']
    require(Path(command[1]).resolve() == root/'source/evaluate_scalar_precision.py', 'Wrong executed evaluator')
    for name, expected in (('--plan', root/'plan.json'), ('--export', args.export),
                           ('--models', args.models), ('--features', args.features), ('--output', output)):
        require(Path(command[command.index(name)+1]).resolve() == expected.resolve(), 'Launch input differs: '+name)
    for name, digest in plan['source_sha256'].items():
        require(sha(root/'source'/name) == launch['source_sha256'][name] == digest,
                'Executed source differs: '+name)
    for name, digest in plan['parent_files_sha256'].items():
        require(sha(args.models/name) == digest, 'Original INT8 evidence differs: '+name)
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    require(sha(args.features/'features.bin') == plan['feature_data_sha256'], 'Scalar feature bytes differ')
    for name, digest in plan['continuous_feature_sha256'].items():
        require(sha(args.features/name) == digest, 'Continuous feature bytes differ: '+name)
    require(sha(args.export/'result.json') == plan['export_result_sha256'], 'Export result differs')
    smoke = json.loads((args.export/'result.json').read_bytes())
    require(smoke['advance_to_precision_diagnostic'], 'Numerical smoke failed')
    for arm, models in plan['model_files_sha256'].items():
        require(sha(args.export/f'{arm}-float.tflite') == models['float']
                and sha(args.models/f'model-{arm}/hey_nyx.tflite') == models['int8'], 'Model bytes differ')
    sys.path.insert(0, str(root/'source'))
    from evaluate import events
    from evaluate_compact_sequence import summarize
    from evaluate_scalar_precision import paired_counts, strict_point
    rows = json.loads((args.features/'features.json').read_bytes())
    selected = [row for row in rows if row['split'] == 'validation']
    reserved = {row['sha256'] for row in rows if row['split'] != 'validation'}
    require(all(row['sha256'] not in reserved for row in selected), 'Validation source overlaps another split')
    ambient_rows = json.loads((args.features/'human.json').read_bytes())
    ambient_rows += json.loads((args.features/'ambient.json').read_bytes())
    selected_ambient = [(i, row) for i, row in enumerate(ambient_rows) if row['split'] == 'validation']
    result = json.loads((output/'result.json').read_bytes())
    require(result['accepted'] is False and result['plan_sha256'] == sha(root/'plan.json'), 'Result binding differs')
    reports = {}
    for arm in plan['model_files_sha256']:
        measured = result['arms'][arm]
        require(measured['models_sha256'] == plan['model_files_sha256'][arm], 'Model binding differs')
        original = json.loads((args.models/f'validation-{arm}/validation-details.json').read_bytes())
        clips = {p: [] for p in ('float', 'int8')}
        ambient = {p: [] for p in clips}
        paired, clip_steps, continuous_steps = [], 0, 0
        delta_sum, delta_max, delta_steps = 0., 0., 0
        with (output/f'clips-{arm}.jsonl').open() as stream:
            for row, cached, line in zip(selected, original['validation'], stream, strict=True):
                record = json.loads(line)
                require(record['sha256'] == row['sha256'], 'Clip source hash differs')
                arrays, maxima = {}, {}
                for precision in clips:
                    flat = {**{k: record[k] for k in ('source', 'engine', 'variant', 'label', 'text', 'times', 'keyword_end')},
                            **record['arms'][precision]}
                    clip, count = reconstruct_clip(row, flat)
                    clips[precision].append(clip)
                    arrays[precision] = np.asarray(flat['scores'])
                    maxima[precision] = {'max_score': clip['max_score']}
                require(np.array_equal(arrays['int8'], cached['scores']), 'Retained INT8 clip replay differs')
                paired.append({'label': row['label'], 'arms': maxima})
                delta = np.abs(arrays['float']-arrays['int8'])
                delta_sum += float(delta.sum()); delta_max = max(delta_max, float(delta.max())); delta_steps += count
                clip_steps += count
        details = json.loads((output/f'ambient-{arm}.json').read_bytes())
        arrays_report = {}
        for (index, row), item, cached in zip(selected_ambient, details, original['ambient'], strict=True):
            source = np.load(args.features/row['path'], allow_pickle=False, mmap_mode='r')
            count = len(source)//3-4
            hours = len(source)*.01/3600
            source_type = row.get('source_type', 'speech' if 'speaker' in row else 'environment')
            require(item['feature_path'] == row['path']
                    and item['feature_sha256'] == plan['continuous_feature_sha256'][row['path']]
                    and item['hours'] == hours and item['source_type'] == source_type,
                    'Continuous source/exposure differs')
            arrays = {}
            for precision in clips:
                entry = item['arms'][precision]
                name = f'ambient-{arm}-{precision}-{index:03d}.npy'
                require(entry['array'] == name and entry['sha256'] == sha(output/name), 'Trace file differs')
                values = np.load(output/name, allow_pickle=False)
                require(values.shape == (count,) and np.isfinite(values).all()
                        and np.all((values >= 0) & (values <= 1)), 'Invalid continuous values')
                counts = {str(t): events(values, t) for t in plan['threshold_grid']}
                require(counts == entry['events'], 'Continuous events do not reconstruct')
                ambient[precision].append({'hours': hours, 'source_type': source_type, 'events': counts})
                arrays[precision] = values
                arrays_report[name] = entry['sha256']
            original_array = args.models/f'validation-{arm}/ambient-validation-{index:03d}.npy'
            require(np.array_equal(arrays['int8'], np.load(original_array, allow_pickle=False))
                    and ambient['int8'][-1] == cached, 'Retained continuous INT8 replay differs')
            delta = np.abs(arrays['float']-arrays['int8'])
            delta_sum += float(delta.sum()); delta_max = max(delta_max, float(delta.max())); delta_steps += count
            continuous_steps += count
        grids = {p: [summarize(clips[p], ambient[p], t) for t in plan['threshold_grid']] for p in clips}
        require(equal(grids, measured['grids']), 'Full precision grids do not reconstruct')
        require(equal({p: strict_point(grid) for p, grid in grids.items()}, measured['strict_gate']),
                'Strict-gate existence does not reconstruct')
        require(paired_counts(paired, .97) == measured['paired_views_at_reference_0_97'],
                'Paired precision contingency differs')
        require(all(measured[k] == v for k, v in {
            'clip_count': len(selected), 'clip_smoothed_steps': clip_steps,
            'continuous_source_count': len(selected_ambient), 'continuous_smoothed_steps': continuous_steps,
            'all_smoothed_steps': delta_steps, 'max_absolute_precision_delta': delta_max,
            'mean_absolute_precision_delta': delta_sum/delta_steps}.items()), 'Precision exposure/error differs')
        reports[arm] = {'clips': len(selected), 'clip_steps': clip_steps,
            'continuous_sources': len(selected_ambient), 'continuous_steps': continuous_steps,
            'clip_trace_sha256': sha(output/f'clips-{arm}.jsonl'), 'continuous_arrays': arrays_report}
    return {'source_sha256': sha(Path(__file__)), 'reconstruction_helper_sha256': sha(Path(__file__).with_name('audit_scalar_wake_scores.py')),
        'plan_sha256': sha(root/'plan.json'), 'result_sha256': sha(output/'result.json'),
        'frozen_plan_source_parent_and_feature_bytes_match': True,
        'all_maxima_timebases_events_grids_contingencies_and_errors_reconstructed': True,
        'all_retained_int8_scores_match_original': True, 'arms': reports,
        'limits': 'Read-only retained-score audit, not a second inference replay. The frozen evaluator separately requires fresh INT8 to reproduce the original traces. No independent recognition, physical latency, resource, pronunciation or voice stability acceptance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'export', 'models', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'arms'}))


if __name__ == '__main__':
    main()
