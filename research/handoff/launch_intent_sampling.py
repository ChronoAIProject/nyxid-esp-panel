"""Freeze and run the committed training-only word sampling comparison."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from audit_compact_study import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); root, repo = args.artifacts.resolve(), args.repository.resolve()
    relative = 'research/wakeword_p4/training/plans/intent-source-balanced-sampling-v1.json'
    plan_path = repo/relative; plan = json.loads(plan_path.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    assert subprocess.check_output(['git', 'show', revision+':'+relative], cwd=repo) == plan_path.read_bytes()
    for name, digest in plan['source_sha256'].items():
        relative_source = 'research/wakeword_p4/training/'+name
        assert sha(repo/relative_source) == digest
        assert subprocess.check_output(['git', 'show', revision+':'+relative_source], cwd=repo) == (repo/relative_source).read_bytes()
    assert sha(Path(__file__)) == plan['launcher_sha256']
    root.mkdir(parents=True, exist_ok=False); (root/'source').mkdir()
    for path in (repo/'research/wakeword_p4/training').glob('*.py'):
        shutil.copyfile(path, root/'source'/path.name)
    shutil.copyfile(plan_path, root/'plan.json'); shutil.copyfile(Path(__file__), root/'launcher.py')
    flags = {'PYTHONHASHSEED': '0', 'TF_ENABLE_ONEDNN_OPTS': '0', 'TF_CPP_MIN_LOG_LEVEL': '2', 'PYTHONUNBUFFERED': '1'}
    commands = {mode: [sys.executable, str(root/'source/study_intent_sampling.py'), '--mode', mode,
        '--artifacts', str(root), '--upstream', str(args.upstream.resolve()),
        '--features', str(args.features.resolve()), '--bank', str(args.bank.resolve()),
        '--initial-weights', str(args.initial_weights.resolve())] for mode in ('train', 'audit')}
    launch = {'checkout_revision': revision, 'started_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(root/'plan.json'), 'commands': commands, 'environment_flags': flags,
        'runtime': {'python': sys.version, **{k: version(k) for k in ('tensorflow', 'numpy', 'keras', 'h5py')}},
        'source_sha256': {p.name: sha(p) for p in (root/'source').glob('*.py')},
        'scope': 'Silent cached training features only; no audio devices or hardware.'}
    (root/'launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    for mode, command in commands.items():
        with (root/(mode+'.log')).open('xb') as log:
            completed = subprocess.run(command, env={**os.environ, **flags}, stdout=log, stderr=subprocess.STDOUT)
        (root/(mode+'-exit.json')).write_text(json.dumps({'returncode': completed.returncode})+'\n')
        if completed.returncode:
            raise RuntimeError(mode+' failed; retain the original evidence')
        print(json.dumps({'phase': mode, 'returncode': completed.returncode}), flush=True)


if __name__ == '__main__':
    main()
