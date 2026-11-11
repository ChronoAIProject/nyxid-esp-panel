"""Audit background supervision, replay the prior CTC control, score treatment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from audit_compact_study import audit, require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights', 'prior-study'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.plan_repository_path = 'research/wakeword_p4/training/plans/microwake-background-ctc-v1.json'
    root, prior = args.artifacts.resolve(), args.prior_study.resolve()
    plan = json.loads((root/'plan.json').read_bytes())
    for name, digest in plan['postprocessing_source_sha256'].items():
        require(sha(Path(__file__).parent/name) == digest, 'Postprocessing source differs: '+name)
    for name, digest in plan['prior_control_files_sha256'].items():
        require(sha(prior/name) == digest, 'Prior control evidence differs: '+name)
    checked = audit(args)
    with (root/'input-audit.json').open('x') as out:
        out.write(json.dumps(checked, indent=2)+'\n')
    print(json.dumps({'phase': 'input_audit', 'passed': True}), flush=True)
    source = root/'model-phonetic/source'
    require(sha(source/'evaluate_stream.py') == sha(prior/'model-phonetic/source/evaluate_stream.py'),
            'Original control evaluator differs')
    original = prior/'validation-phonetic'
    calibration = json.loads((original/'threshold.json').read_bytes())
    require(calibration['model_sha256'] == sha(root/'model-control/hey_nyx.tflite')
            and calibration['threshold_grid'] == plan['threshold_grid']
            and calibration['feature_index_sha256'] == plan['feature_index_sha256']['features.json'],
            'Replayed control calibration identity differs')
    destination = root/'validation-control'
    destination.mkdir(exist_ok=False)
    retained = {}
    for relative, digest in plan['prior_control_files_sha256'].items():
        path = Path(relative)
        if path.parent.as_posix() == 'validation-phonetic':
            require(path.suffix in ('.npy', '.json'), 'Unexpected control score artifact')
            shutil.copyfile(prior/path, destination/path.name)
            require(sha(destination/path.name) == digest, 'Control score copy differs')
            retained[path.name] = digest
    with (root/'control-score-reuse.json').open('x') as out:
        out.write(json.dumps({'policy': 'Prior compact-CTC development scores reused only after exact scalar model, legacy input prefix, PTQ input, warmup and pair-schedule replay.',
            'prior_directory': str(original), 'files_sha256': retained,
            'input_audit_sha256': sha(root/'input-audit.json')}, indent=2)+'\n')
    command = [sys.executable, str(source/'evaluate_stream.py'),
        '--model', str(root/'model-phonetic/hey_nyx.tflite'), '--features', str(args.features.resolve()),
        '--output', str(root/'validation-phonetic'), *plan['validation_args'],
        '--threshold-grid', *map(str, plan['threshold_grid'])]
    with (root/'validation-phonetic-launch.json').open('x') as out:
        out.write(json.dumps({'command': command, 'plan_sha256': sha(root/'plan.json'),
            'evaluator_sha256': sha(source/'evaluate_stream.py')}, indent=2)+'\n')
    with (root/'validation-phonetic.log').open('xb') as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'treatment_validation', 'passed': True}), flush=True)
    command = [sys.executable, str(source/'compare_phone_distillation.py'),
        '--plan', str(root/'plan.json'), '--artifacts', str(root), '--output', str(root/'comparison.json')]
    with (root/'comparison-launch.json').open('x') as out:
        out.write(json.dumps({'command': command}, indent=2)+'\n')
    with (root/'comparison.log').open('xb') as out:
        subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'complete',
        'advance': json.loads((root/'comparison.json').read_bytes())['advance']}), flush=True)


if __name__ == '__main__':
    main()
