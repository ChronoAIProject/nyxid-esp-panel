"""Audit the frozen compact study, reuse its exact control, then score treatment.

Optional waiting watches artifact completion only. It accesses no audio device
or firmware. A failed audit stops before treatment validation.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_compact_study import audit, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights', 'prior-control'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    args.plan_repository_path = 'research/wakeword_p4/training/plans/microwake-compact-phrase-ctc-v1.json'
    root = args.artifacts.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    prior = args.prior_control.resolve()
    source = root/'model-phonetic/source'
    evaluator = source/'evaluate_stream.py'
    comparator = source/'compare_phone_distillation.py'
    previous_source = prior/'model-control/source'
    for name in ('evaluate_stream.py', 'evaluate.py', 'features.py', 'detector_policy.py'):
        require(sha(source/name) == sha(previous_source/name) == plan['source_sha256'][name],
                'Control evaluation source differs: '+name)
    previous_cal = json.loads((prior/'validation-control/threshold.json').read_bytes())
    require(previous_cal['model_sha256'] == plan['required_control_replay']['model_sha256']
            and previous_cal['feature_index_sha256'] == plan['feature_index_sha256']['features.json']
            and previous_cal['threshold_grid'] == plan['threshold_grid'], 'Control calibration input differs')
    command = [sys.executable, str(evaluator), '--model', str(root/'model-phonetic/hey_nyx.tflite'),
        '--features', str(args.features.resolve()), '--output', str(root/'validation-phonetic'),
        *plan['validation_args'], '--threshold-grid', *map(str, plan['threshold_grid'])]
    retained = {p.name: sha(p) for p in sorted((prior/'validation-control').iterdir()) if p.is_file()}
    provenance = {'source_sha256': sha(Path(__file__)), 'study_plan_sha256': sha(root/'plan.json'),
        'control_policy': 'Reuse byte-exact prior development scores only after original control export/input replay passes.',
        'prior_control_directory': str(prior/'validation-control'), 'prior_control_files_sha256': retained,
        'treatment_validation_command': command, 'evaluator_sha256': sha(evaluator),
        'comparison_source_sha256': sha(comparator),
        'scope': 'Unchanged development-only grid/gates; no final scores, threshold extension or board operation.'}
    with (root/'postprocess-plan.json').open('x') as stream:
        stream.write(json.dumps(provenance, indent=2)+'\n')
    deadline = time.monotonic()+3600
    while True:
        try:
            for arm in plan['arms']:
                json.loads((root/f'model-{arm}/training.json').read_bytes())
            break
        except (FileNotFoundError, json.JSONDecodeError):
            if not args.wait:
                raise
            for arm in plan['arms']:
                log = root/f'train-{arm}.log'
                require('Traceback (most recent call last)' not in log.read_text(), 'Training failed: '+arm)
            require(time.monotonic() < deadline, 'Training completion timeout')
            time.sleep(10)
    checked = audit(args)
    with (root/'input-audit.json').open('x') as stream:
        stream.write(json.dumps(checked, indent=2)+'\n')
    print(json.dumps({'phase': 'input_audit', 'passed': True}), flush=True)
    target = root/'validation-control'
    target.mkdir(exist_ok=False)
    for name, digest in retained.items():
        source_file = prior/'validation-control'/name
        require(source_file.suffix in ('.json', '.npy') and sha(source_file) == digest, 'Control artifact changed')
        shutil.copyfile(source_file, target/name)
        require(sha(target/name) == digest, 'Copied control artifact differs')
    with (root/'validation-phonetic.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'treatment_validation', 'passed': True}), flush=True)
    comparison = [sys.executable, str(comparator), '--plan', str(root/'plan.json'),
        '--artifacts', str(root), '--output', str(root/'comparison.json')]
    with (root/'comparison.log').open('xb') as log:
        subprocess.run(comparison, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'complete', 'advance': json.loads((root/'comparison.json').read_text())['advance']}), flush=True)


if __name__ == '__main__':
    main()
