"""Launch a frozen compact CTC comparison from immutable source copies."""
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
    for name in ('artifacts', 'repository', 'upstream', 'features', 'bank', 'initial-weights',
                 'prior-study', 'background-inputs'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--plan', type=Path,
        default=Path('research/wakeword_p4/training/plans/microwake-positive-ctc-v1.json'))
    args = parser.parse_args()
    repo, root = args.repository.resolve(), args.artifacts.resolve()
    plan_source = (repo/args.plan).resolve()
    plan_relative = plan_source.relative_to(repo).as_posix()
    plan = json.loads(plan_source.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    require(subprocess.check_output(['git', 'show', revision+':'+plan_relative], cwd=repo)
            == plan_source.read_bytes(), 'Plan is not committed')
    for name, digest in plan['source_sha256'].items():
        require(sha(repo/'research/wakeword_p4/training'/name) == digest, 'Training source differs: '+name)
    for name, digest in plan['postprocessing_source_sha256'].items():
        require(sha(repo/'research/handoff'/name) == digest, 'Postprocessing source differs: '+name)
    for name, digest in plan['prior_control_files_sha256'].items():
        require(sha(args.prior_study/name) == digest, 'Parent evidence differs: '+name)
    require(sha(args.background_inputs) == plan['background_inputs_sha256'], 'Background inventory differs')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_source, root/'plan.json')
    shutil.copyfile(args.background_inputs, root/'background-inputs.json')
    shutil.copyfile(Path(__file__), root/'launcher.py')
    post = root/'postprocessing'
    post.mkdir()
    for name in plan['postprocessing_source_sha256']:
        shutil.copyfile(repo/'research/handoff'/name, post/name)
    flags = {'PYTHONHASHSEED': '0', 'TF_ENABLE_ONEDNN_OPTS': '0',
             'TF_CPP_MIN_LOG_LEVEL': '2', 'PYTHONUNBUFFERED': '1'}
    os.environ.update(flags)
    commands = {}
    for arm, gain in plan['arms'].items():
        commands[arm] = [sys.executable, str(repo/'research/wakeword_p4/training/run_training.py'),
            '--output', str(root/f'model-{arm}'), '--upstream', str(args.upstream.resolve()),
            '--features', str(args.features.resolve()), '--initial-weights', str(args.initial_weights.resolve()),
            '--phone-bank', str(args.bank.resolve()), '--phone-gain', str(gain),
            '--compact-phrase-plan', str(root/'plan.json')]
        for field, option in (('background_gains', '--phone-background-gain'),
                              ('positive_gains', '--phone-positive-gain'),
                              ('intent_gains', '--compact-intent-gain')):
            if arm in plan.get(field, {}):
                commands[arm] += [option, str(plan[field][arm])]
        commands[arm] += plan['common_train_args']
    postcommand = [sys.executable, str(post/'finish_background_ctc_study.py'),
        '--artifacts', str(root), '--repository', str(repo), '--upstream', str(args.upstream.resolve()),
        '--features', str(args.features.resolve()), '--bank', str(args.bank.resolve()),
        '--initial-weights', str(args.initial_weights.resolve()), '--prior-study', str(args.prior_study.resolve()),
        '--plan-repository-path', plan_relative]
    launch = {'checkout_revision': revision, 'plan_sha256': sha(root/'plan.json'),
        'started_utc': datetime.now(timezone.utc).isoformat(), 'commands': commands,
        'postprocess_command': postcommand, 'environment_flags': flags,
        'runtime': {'python': sys.version, **{k: version(k) for k in
            ['tensorflow', 'numpy', 'keras', 'h5py', 'scipy', 'ai-edge-litert']}},
        'launcher_sha256': sha(root/'launcher.py'),
        'scope': 'Silent cached-feature research. No audio device or board access. No environment secrets copied.'}
    (root/'study-launch.json').write_text(json.dumps(launch, indent=2)+'\n')
    processes, logs = {}, {}
    for arm, command in commands.items():
        logs[arm] = (root/f'train-{arm}.log').open('xb')
        processes[arm] = subprocess.Popen(command, cwd=repo, stdout=logs[arm], stderr=subprocess.STDOUT)
    print(json.dumps({'phase': 'training_launched', 'pids': {k: p.pid for k, p in processes.items()},
                      'revision': revision}), flush=True)
    statuses = {arm: process.wait() for arm, process in processes.items()}
    for log in logs.values():
        log.close()
    (root/'training-exit.json').write_text(json.dumps(statuses, indent=2)+'\n')
    require(all(code == 0 for code in statuses.values()), 'Training failed: '+str(statuses))
    print(json.dumps({'phase': 'training_complete', 'statuses': statuses}), flush=True)
    with (root/'postprocess.log').open('xb') as log:
        subprocess.run(postcommand, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'phase': 'complete', 'advance':
                     json.loads((root/'comparison.json').read_bytes())['advance']}), flush=True)


if __name__ == '__main__':
    main()
