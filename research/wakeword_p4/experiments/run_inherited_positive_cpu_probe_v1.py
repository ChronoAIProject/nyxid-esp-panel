"""Run six isolated CPU captures; never touches the existing GPU formal run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


BASE_SHA = '2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904'
PATCHED_SHA = 'b97b2debc92ca106ab199af60ac144aa7a78ce4a26e39621bc5b179fb637917e'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--control-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    cli = json.loads(args.control_config.read_text())['trainer_args']
    assert all(isinstance(x, str) for x in cli) and '--output' not in cli
    assert '--recover-inherited-positive-draws' not in cli
    expected = {'--steps': '120000', '--eval-interval': '5000', '--context-frames': '150',
                '--architecture': 'teacher60', '--batchnorm-policy': 'frozen',
                '--paired-objective': 'phrase-ranking', '--paired-rendering-coefficient': '1',
                '--paired-draw-noise-condition': 'fixed'}
    for flag, value in expected.items():
        assert cli.count(flag) == 1 and cli[cli.index(flag) + 1] == value
    # Common CPU probe policy; no formal JSON or executable source is edited.
    cli[cli.index('--device') + 1] = 'cpu'
    cli[cli.index('--threads') + 1] = '2'
    cli = [x for x in cli if x != '--recover-erased-complete-coverage']
    root = args.output.resolve()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref + '^{commit}'], text=True).strip()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit,
                                     'research/wakeword_p4/training'], text=True).splitlines()
    sources = root / 'base-source'
    for name in names:
        if name.endswith('.py'):
            destination = sources / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(['git', 'show', commit + ':' + name]))
    relative = 'research/wakeword_p4/training/train_stream.py'
    assert sha(sources / relative) == BASE_SHA
    patched = root / 'patched-source'
    shutil.copytree(sources, patched)
    tools = Path(__file__).resolve().parent
    patch = tools / 'recover_inherited_positive_draws_v1.patch'
    subprocess.run(['git', 'apply', '--check', str(patch)], cwd=patched, check=True)
    subprocess.run(['git', 'apply', str(patch)], cwd=patched, check=True)
    assert sha(patched / relative) == PATCHED_SHA
    completed = []
    for complete in (False, True):
        case = root / ('complete-on' if complete else 'complete-off')
        case.mkdir()
        case_cli = cli + (['--recover-erased-complete-coverage'] if complete else [])
        control = case / 'control.json'
        enabled = case / 'enabled.json'
        control.write_text(json.dumps(dict(trainer_args=case_cli), indent=2) + '\n')
        enabled.write_text(json.dumps(dict(trainer_args=case_cli + ['--recover-inherited-positive-draws']), indent=2) + '\n')
        for name, source, config in (('original', sources, control), ('disabled', patched, control),
                                     ('enabled', patched, enabled)):
            print(json.dumps(dict(case=case.name, arm=name, starting=True)), flush=True)
            with (case / (name + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-u', str(tools / 'capture_inherited_positive_integration_v1.py'),
                                '--trainer', str(source / relative), '--config', str(config),
                                '--output', str(case / name)], stdout=log, stderr=subprocess.STDOUT, check=True)
        with (case / 'comparison.log').open('w') as log:
            subprocess.run([sys.executable, str(tools / 'compare_inherited_positive_integration_v1.py'),
                            '--original', str(case / 'original'), '--disabled', str(case / 'disabled'),
                            '--enabled', str(case / 'enabled'), '--output', str(case / 'comparison.json')],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        completed.append(json.loads((case / 'comparison.json').read_text()))
        print(json.dumps(dict(case=case.name, all_three_real_two_update_comparisons_passed=True)), flush=True)
    report = dict(schema='inherited-positive-cpu-probe-sequence-v1',
        source_sha256=sha(Path(__file__)), gpu_commit=commit,
        original_config_sha256=sha(args.control_config), common_probe_changes='device=cpu, threads=2 only',
        cases=completed, total_optimizer_updates=12, existing_gpu_sequence_modified=False,
        acceptance_claim=False)
    (root / 'sequence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cpu_integration_complete=True, acceptance_claim=False)), flush=True)


if __name__ == '__main__':
    main()
