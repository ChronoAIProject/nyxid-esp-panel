"""Freeze matched measured-room means, then run the original offline regression.

Run calibrate first. Preserve the generated plans before running regress.
Only original validation and existing development inputs are used; no playback.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / 'research/wakeword_p4/embedding'
PLANS = CODE / 'plans'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def run(logs, name, command):
    print(json.dumps({'job': name, 'starting': True}), flush=True)
    with (logs / (name + '.log')).open('x') as stream:
        subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'job': name, 'complete': True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['calibrate', 'regress'])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--log-output', type=Path, required=True)
    args = parser.parse_args()
    data, logs = args.data_root.resolve(), args.log_output.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    manifest_path = PLANS / 'measured-room-continuation-v1.json'
    manifest = read(manifest_path)
    original_regression = read(PLANS / 'head-equal-mean-eq-regression-plan-v1.json')
    mean_template = read(PLANS / 'head-equal-mean-eq-plan-v1.json')
    members = {}
    for entry in manifest['members']:
        training, plan_path = data / entry['output'], PLANS / entry['plan']
        plan = read(plan_path)
        if (sha(plan_path) != entry['plan_sha256']
                or read(training / 'complete.json')['plan_sha256'] != entry['plan_sha256']
                or plan['event_policy'] != mean_template['event_policy']
                or plan['selection_policy'] != mean_template['selection_policy']):
            raise ValueError('Training identity or frozen validation policy differs')
        selection = read(training / 'checkpoint-selection.json')
        if selection['selected_step'] is None:
            raise ValueError('A member has no validation-selected checkpoint')
        root = training / f"step-{selection['selected_step']:05d}"
        if sha(root / 'head.tflite') != selection['selected_head_sha256']:
            raise ValueError('Selected checkpoint identity differs')
        member = {
            'checkpoint': str(root.relative_to(data)),
            'head_sha256': sha(root / 'head.tflite'),
            'calibration_sha256': sha(root / 'validation/calibration.json'),
            'details_sha256': sha(root / 'validation/details-private.json'),
            'continuous_scores': {p.name: sha(p) for p in sorted((root / 'validation').glob('*-scores.npy'))},
        }
        key = entry['arm'], entry['head']
        if key in members:
            raise ValueError('Duplicate arm/head')
        members[key] = member

    for arm in ['control', 'measured']:
        name = f'head-equal-mean-measured-room-{arm}-v1'
        plan_path = PLANS / f'{name}-plan.json'
        regression_path = PLANS / f'{name}-regression-plan.json'
        output = data / name
        selected_members = [members[(arm, head)] for head in ['dense', 'temporal']]
        if args.stage == 'calibrate':
            mean = {
                'scope': f'Prespecified validation-only equal mean, matched measured-room {arm}. '
                         'No fitted weights, microphone/development/final choice.',
                'comparison_plan_sha256': sha(manifest_path),
                'members': selected_members,
                'combination': mean_template['combination'],
                'selection_policy': mean_template['selection_policy'],
                'event_policy': mean_template['event_policy'],
            }
            save(plan_path, mean)
            run(logs, name, [sys.executable, str(CODE / 'evaluate_equal_mean.py'),
                '--data-root', str(data), '--plan', str(plan_path), '--output', str(output)])
            result = read(output / 'calibration.json')
            if result['plan_sha256'] != sha(plan_path) or result['selected'] is None:
                raise ValueError('Equal mean lacks a matching validation selection')
            regression = copy.deepcopy(original_regression)
            regression.update({
                'scope': f'Validation-frozen matched measured-room {arm} development regression. '
                         'Original unchanged PCM/cached features only; no threshold selection or final/physical acceptance.',
                'arm': arm,
                'comparison_plan_sha256': sha(manifest_path),
                'selection_sha256': sha(output / 'calibration.json'),
                'selected': {
                    'members': result['members'], 'selected': result['selected'],
                    'event_policy': mean['event_policy'], 'model_sha256': sha(output / 'calibration.json'),
                },
            })
            save(regression_path, regression)
            print(json.dumps({'arm': arm, 'regression_plan_sha256': sha(regression_path),
                'selection': {key: result['selected'][key] for key in
                    ('threshold', 'macro_recall', 'continuous_events', 'max_corpus_false_rate')}}), flush=True)
        else:
            frozen = read(regression_path)
            calibrated = read(output / 'calibration.json')
            if (frozen['selection_sha256'] != sha(output / 'calibration.json')
                    or calibrated['plan_sha256'] != sha(plan_path)
                    or frozen['comparison_plan_sha256'] != sha(manifest_path)
                    or frozen['selected']['members'] != selected_members
                    or frozen['selected']['selected'] != calibrated['selected']
                    or frozen['selected']['event_policy'] != mean_template['event_policy']):
                raise ValueError('Frozen regression differs from the validation selection')
            for kind in ['paired', 'english', 'mandarin']:
                target = data / f'measured-room-{arm}-{kind}-v1'
                run(logs, target.name, [sys.executable, str(CODE / 'regress_equal_mean.py'),
                    '--data-root', str(data), '--plan', str(regression_path), '--kind', kind,
                    '--output', str(target), '--workers', '3'])
                print(json.dumps({'arm': arm, 'kind': kind, 'report': read(target / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
