"""Audit the new broader-bank control and treatment, then score both silently."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from audit_compact_study import audit, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.plan_repository_path = 'research/wakeword_p4/training/plans/microwake-compact-broad-ctc-v1.json'
    root = args.artifacts.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    for name, digest in plan['postprocessing_source_sha256'].items():
        require(sha(Path(__file__).parent/name) == digest, 'Postprocessing source differs: '+name)
    checked = audit(args)
    with (root/'input-audit.json').open('x') as out:
        out.write(json.dumps(checked, indent=2)+'\n')
    print(json.dumps({'phase': 'input_audit', 'passed': True}), flush=True)
    for arm in plan['arms']:
        source = root/f'model-{arm}/source'
        command = [sys.executable, str(source/'evaluate_stream.py'),
            '--model', str(root/f'model-{arm}/hey_nyx.tflite'),
            '--features', str(args.features.resolve()), '--output', str(root/f'validation-{arm}'),
            *plan['validation_args'], '--threshold-grid', *map(str, plan['threshold_grid'])]
        with (root/f'validation-{arm}-launch.json').open('x') as out:
            out.write(json.dumps({'command': command, 'plan_sha256': sha(root/'plan.json'),
                'evaluator_sha256': sha(source/'evaluate_stream.py')}, indent=2)+'\n')
        with (root/f'validation-{arm}.log').open('xb') as out:
            subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({'phase': 'validation', 'arm': arm, 'passed': True}), flush=True)
    comparison = [sys.executable, str(root/'model-phonetic/source/compare_phone_distillation.py'),
        '--plan', str(root/'plan.json'), '--artifacts', str(root), '--output', str(root/'comparison.json')]
    with (root/'comparison-launch.json').open('x') as out:
        out.write(json.dumps({'command': comparison}, indent=2)+'\n')
    with (root/'comparison.log').open('xb') as out:
        subprocess.run(comparison, stdout=out, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'complete',
        'advance': json.loads((root/'comparison.json').read_bytes())['advance']}), flush=True)


if __name__ == '__main__':
    main()
