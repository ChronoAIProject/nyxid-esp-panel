"""Launch a frozen, training-only shared-phone representation diagnostic."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_compact_study import require, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    repo, root = args.repository.resolve(), args.artifacts.resolve()
    relative = 'research/wakeword_p4/training/plans/intent-phone-generalization-v1.json'
    plan_path = repo/relative
    plan = json.loads(plan_path.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    require(subprocess.check_output(['git', 'show', revision+':'+relative], cwd=repo)
            == plan_path.read_bytes(), 'Plan is not committed')
    for name, digest in plan['source_sha256'].items():
        require(sha(repo/'research/wakeword_p4/training'/name) == digest, 'Training source differs: '+name)
    for name, digest in plan['audit_source_sha256'].items():
        require(sha(repo/'research/handoff'/name) == digest, 'Audit source differs: '+name)
    root.mkdir(parents=True, exist_ok=False)
    source, auditing = root/'source', root/'auditing'
    source.mkdir(); auditing.mkdir()
    for path in (repo/'research/wakeword_p4/training').glob('*.py'):
        shutil.copyfile(path, source/path.name)
    for name in plan['audit_source_sha256']:
        shutil.copyfile(repo/'research/handoff'/name, auditing/name)
    shutil.copyfile(plan_path, root/'plan.json')
    shutil.copyfile(Path(__file__), root/'launcher.py')
    command = [sys.executable, str(source/'intent_phone_fit.py'), '--plan', str(root/'plan.json'),
        '--upstream', str(args.upstream.resolve()), '--features', str(args.features.resolve()),
        '--bank', str(args.bank.resolve()), '--initial-weights', str(args.initial_weights.resolve()),
        '--output', str(root/'study')]
    audit_command = [sys.executable, str(auditing/'audit_intent_phone_fit.py'),
        '--artifacts', str(root), '--repository', str(repo), '--features', str(args.features.resolve()),
        '--bank', str(args.bank.resolve()), '--initial-weights', str(args.initial_weights.resolve()),
        '--output', str(root/'input-result-audit.json')]
    flags = {'PYTHONHASHSEED': '0', 'TF_ENABLE_ONEDNN_OPTS': '0',
             'TF_CPP_MIN_LOG_LEVEL': '2', 'PYTHONUNBUFFERED': '1'}
    os.environ.update(flags)
    launch = {'checkout_revision': revision, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(root/'plan.json'), 'command': command, 'audit_command': audit_command,
        'source_sha256': {p.name: sha(p) for p in source.glob('*.py')},
        'launcher_sha256': sha(root/'launcher.py'), 'environment_flags': flags,
        'runtime': {'python': sys.version, **{k: version(k) for k in ('tensorflow', 'numpy', 'keras', 'h5py')}},
        'scope': 'Silent cached training-feature diagnostic; no audio device or hardware access.'}
    (root/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    print(json.dumps({'phase': 'launched', 'revision': revision}), flush=True)
    with (root/'training.log').open('xb') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    (root/'training-exit.json').write_text(json.dumps({'returncode': completed.returncode})+'\n')
    require(completed.returncode == 0, 'Diagnostic training failed')
    with (root/'audit.log').open('xb') as log:
        subprocess.run(audit_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((root/'input-result-audit.json').read_bytes())
    print(json.dumps({'phase': 'complete', 'advance': result['advance'],
        'engine_macro_accuracy': result['engine_macro_accuracy']}), flush=True)


if __name__ == '__main__':
    main()
