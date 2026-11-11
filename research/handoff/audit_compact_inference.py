"""Reconstruct frozen compact inference metrics from retained score traces.

Read-only cached-score audit; no new model inference, training or audio access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(args):
    root, repo = args.artifacts.resolve(), args.repository.resolve()
    output = root/'evaluation'
    plan = json.loads((root/'plan.json').read_bytes())
    launch = json.loads((root/'launch.json').read_bytes())
    revision = launch['checkout_revision']
    plan_path = getattr(args, 'plan_repository_path',
        'research/wakeword_p4/training/plans/compact-sequence-inference-v1.json')
    committed = subprocess.check_output(['git', 'show', revision+':'+plan_path], cwd=repo)
    require(hashlib.sha256(committed).hexdigest() == sha(root/'plan.json') == launch['plan_sha256'],
            'Plan differs from frozen commit or launch')
    require(sha(output/'plan.json') == sha(root/'plan.json'), 'Evaluation plan differs')
    command = launch['command']
    require(Path(command[1]).resolve() == root/'source/evaluate_compact_sequence.py',
            'Wrong executed evaluator')
    require(command[2::2] == ['--plan', '--export', '--features', '--output'], 'Evaluation options differ')
    for name, expected in (('--plan', root/'plan.json'), ('--export', args.export),
                           ('--features', args.features), ('--output', output)):
        require(Path(command[command.index(name)+1]).resolve() == expected.resolve(),
                'Evaluation input differs: '+name)
    for name, digest in plan.get('postprocessing_source_sha256', {}).items():
        require(sha(Path(__file__).parent/name) == digest, 'Postprocessing source differs: '+name)
    for name, digest in plan['source_sha256'].items():
        require(sha(root/'source'/name) == launch['source_sha256'][name] == digest,
                'Executed source differs: '+name)
    require(sha(args.export/'result.json') == plan['export_result_sha256'], 'Export result differs')
    require(sha(args.export/'compact_dual.tflite') == plan['model_sha256'], 'Model differs')
    for name, digest in plan['feature_index_sha256'].items():
        require(sha(args.features/name) == digest, 'Feature index differs: '+name)
    if plan.get('feature_data_sha256'):
        require(sha(args.features/'features.bin') == plan['feature_data_sha256'], 'Scalar feature bytes differ')
    if plan.get('background_inputs_sha256'):
        from background_inputs import inventory
        require(sha(root/'background-inputs.json') == plan['background_inputs_sha256']
                and inventory(args.features) == json.loads((root/'background-inputs.json').read_bytes()),
                'Actual background inputs differ')
    sys.path.insert(0, str(root/'source'))
    from evaluate_compact_sequence import summarize
    from compare_phone_distillation import operating_points
    from evaluate import events
    result = json.loads((output/'result.json').read_bytes())
    require(result['plan_sha256'] == sha(root/'plan.json') and result['model_sha256'] == plan['model_sha256'],
            'Result binding differs')
    require(result.get('ctc_event', 'exact') == plan.get('ctc_event', 'exact'), 'CTC event differs')
    prior = getattr(args, 'prior_inference', None)
    if plan.get('binary_replay_parent_files_sha256'):
        require(prior is not None, 'Missing binary replay parent')
        prior = prior.resolve()
        for name, digest in plan['binary_replay_parent_files_sha256'].items():
            require(sha(prior/name) == digest, 'Binary replay parent differs: '+name)
    index = json.loads((args.features/'features.json').read_bytes())
    selected = [r for r in index if r['split'] == 'validation']
    reserved = {r['sha256'] for r in index if r['split'] != 'validation'}
    clips = {arm: [] for arm in ('control', 'phonetic')}
    clip_steps = 0
    with (output/'clip-scores.jsonl').open() as stream:
        for row, line in zip(selected, stream, strict=True):
            record = json.loads(line)
            require(row['sha256'] not in reserved, 'Validation source overlaps a reserved split')
            require(all(record[k] == row[k] for k in ('engine', 'variant', 'label', 'text', 'sha256'))
                    and record['source'] == row['path'], 'Clip identity differs')
            count = row['frames']//3-4
            times = (3*(np.arange(count)+4)+5)*.01
            end = (row['keyword_frame']+2)*.01 if row['label'] else None
            require(record['keyword_end'] == end and np.array_equal(record['times'], times),
                    'Clip timebase differs')
            mask = ((times >= end-.1) & (times <= end+.8) if end is not None else np.ones(count, bool))
            arrays = {}
            for arm in clips:
                item = record['arms'][arm]
                values = np.asarray(item['scores'])
                require(values.shape == times.shape and np.isfinite(values).all(), 'Invalid clip scores')
                maximum = float(values[mask].max()) if mask.any() else 0.
                require(maximum == item['max_score'], 'Clip maximum does not reproduce')
                clips[arm].append({'engine': row['engine'], 'source': row['path'],
                                  'label': row['label'], 'max_score': maximum})
                arrays[arm] = values
            require(np.all(arrays['phonetic'] >= 0) and np.all(arrays['phonetic'] <= arrays['control']+1e-12),
                    'Product dominance violated')
            clip_steps += count
    ambient = json.loads((output/'ambient-details.json').read_bytes())
    ambient_index = json.loads((args.features/'human.json').read_bytes())
    ambient_index += json.loads((args.features/'ambient.json').read_bytes())
    selected_ambient = [(i, r) for i, r in enumerate(ambient_index) if r['split'] == 'validation']
    require(all(len(ambient[arm]) == len(selected_ambient) for arm in clips), 'Ambient coverage differs')
    event_locations, continuous_steps = {}, 0
    for ordinal, (index, row) in enumerate(selected_ambient):
        source = np.load(args.features/row['path'], allow_pickle=False, mmap_mode='r')
        count, arrays = len(source)//3-4, {}
        for arm in clips:
            item = ambient[arm][ordinal]
            name = f'ambient-{arm}-{index:03d}.npy'
            require(item['feature_path'] == row['path'] and item['array'] == name
                    and item['hours'] == len(source)*.01/3600, 'Ambient source/timebase differs')
            values = np.load(output/name, allow_pickle=False)
            require(values.shape == (count,) and np.isfinite(values).all(), 'Invalid continuous scores')
            require(item['events'] == {str(t): events(values, t) for t in plan['threshold_grid']},
                    'Continuous events do not reproduce')
            arrays[arm] = values
            event_locations[arm+':'+name] = {'source': row['path'], 'score_sha256': sha(output/name),
                                           'events_0_97': item['events']['0.97']}
        require(np.all(arrays['phonetic'] >= 0) and np.all(arrays['phonetic'] <= arrays['control']+1e-12),
                'Continuous product dominance violated')
        continuous_steps += count
    arms = {arm: {'grid': [summarize(clips[arm], ambient[arm], t) for t in plan['threshold_grid']]}
            for arm in clips}
    for arm in arms:
        require(arms[arm]['grid'] == result['arms'][arm]['grid'], 'Grid metrics differ: '+arm)
    require(all(result[k] == v for k, v in operating_points(arms, .97).items()), 'Advancement gate differs')
    if plan.get('binary_replay_parent_files_sha256'):
        with (output/'clip-scores.jsonl').open() as current, (prior/'evaluation/clip-scores.jsonl').open() as original:
            for left, right in zip(current, original, strict=True):
                left, right = json.loads(left), json.loads(right)
                require({k: v for k, v in left.items() if k != 'arms'}
                        == {k: v for k, v in right.items() if k != 'arms'}
                        and left['arms']['control'] == right['arms']['control'],
                        'Same-artifact binary clip replay differs')
        require(ambient['control'] == json.loads((prior/'evaluation/ambient-details.json').read_bytes())['control'],
                'Same-artifact binary continuous metadata differs')
        for item in ambient['control']:
            require(np.array_equal(np.load(output/item['array'], allow_pickle=False),
                                   np.load(prior/'evaluation'/item['array'], allow_pickle=False)),
                    'Same-artifact binary continuous score replay differs')
    return {'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(root/'plan.json'),
        'raw_result_sha256': sha(output/'result.json'), 'launch_sha256': sha(root/'launch.json'),
        'frozen_source_plan_export_and_indices_match': True,
        'clip_count': len(selected), 'clip_smoothed_steps': clip_steps,
        'continuous_source_count': len(selected_ambient), 'continuous_smoothed_steps': continuous_steps,
        'all_maxima_timebases_events_grids_and_gate_reproduced': True,
        'full_scalar_and_background_bytes_hashed': bool(plan.get('feature_data_sha256')
                                                       and plan.get('background_inputs_sha256')),
        'pointwise_product_dominance_verified': True, 'continuous_event_counts_measured_independently': True,
        'parent_binary_scores_replayed_exactly': bool(plan.get('binary_replay_parent_files_sha256')),
        'continuous_arrays': event_locations,
        'limits': 'Cached-score reconstruction only. Full scalar/background bytes are checked only when explicitly hash-bound in the plan; index hashes alone do not bind array bytes. No new inference, independent wake acceptance, pronunciation, physical endpoint, P4 resource or call stability proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'export', 'features', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan-repository-path',
        default='research/wakeword_p4/training/plans/compact-sequence-inference-v1.json')
    parser.add_argument('--prior-inference', type=Path)
    args = parser.parse_args()
    result = audit(args)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'continuous_arrays'}))


if __name__ == '__main__':
    main()
