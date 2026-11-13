"""Separate fresh same-PCM official grid; never mutate the original live grid."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_development_grid import REQUESTS, sha

ROOT = Path('research/wakeword_p4')
PLAN = ROOT/'results/4090-jenny-official-development-execution-v1.json'


def register():
    custom_path = ROOT/'results/4090-jenny-complete-development-execution-v1.json'
    custom = json.loads(custom_path.read_text())
    assert custom['authorizes'] == 'development-raw-collection-only'
    for path, expected in custom['files'].items():
        assert sha(path) == expected, path
    old_path = ROOT/'results/4090-hey-official-development-execution-v1.json'
    old = json.loads(old_path.read_text())
    for path, expected in old['files'].items():
        assert sha(path) == expected, path
    rows = []
    for phase in ('synthetic', 'background'):
        material = custom['materials'][phase]
        assert material['role'] == 'new-development'
        assert sha(material['path']) == material['sha256']
        originals = json.loads(Path(material['path']).read_text())['records']
        assert len(originals) == (396 if phase == 'synthetic' else 12)
        for row in originals:
            item = dict(row, phase=phase)
            if phase == 'background':
                item.update(path=row['wav_path'], sha256=row['wav_sha256'])
            assert sha(item['path']) == item['sha256']
            rows.append(item)
    source = Path(__file__).parent
    runner = source/'run_jenny_development_v1.py'
    paths = [Path(__file__), runner, source/'run_development.py', custom_path, old_path,
             source/'main/main.c']
    paths += [Path(p) for p in old['runtime'].values()]
    paths += [Path(custom['materials'][phase]['path']) for phase in ('synthetic', 'background')]
    files = {str(p): sha(p) for p in paths}
    files.update({r['path']: r['sha256'] for r in rows})
    plan = dict(schema='jenny-official-development-execution-v1', files=files,
                runtime=old['runtime'], runner=str(runner), models=['willow', 'hiesp'],
                threshold_requests=REQUESTS, records=rows, streams=2*len(REQUESTS)*len(rows),
                output='.local/wakeword/jenny-official-development-v1', retries=0,
                authorizes='Official reference complete fresh DEVELOPMENT grid only. Same original PCM as custom collector. No acceptance or promotion.',
                limits='Classic ESP32 QEMU runtime with original bounded certification; no board/P4 equivalence. Jenny single-voice ancestry limits retained; numerical threshold requests are not custom UINT8 thresholds. Raw official results require separately frozen calibration code before selection.')
    with PLAN.open('x') as output:
        output.write(json.dumps(plan, indent=2)+'\n')


def run():
    published = subprocess.check_output(['git', 'show', 'origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert published == PLAN.read_bytes()
    plan = json.loads(published)
    for path, expected in plan['files'].items():
        assert sha(path) == expected, path
    output = Path(plan['output']); output.mkdir(exist_ok=False)
    completed = 0
    for model in plan['models']:
        for config, threshold in enumerate(plan['threshold_requests']):
            for index, row in enumerate(plan['records']):
                case = Path(f'/tmp/jy-dev1-{model}-{config:02}-{index:03}')
                command = [sys.executable, plan['runner']]
                for name, path in plan['runtime'].items():
                    command += ['--'+name, path]
                command += ['--model', model, '--threshold', threshold,
                            '--record-index', str(index), '--output', str(case)]
                stem = f'{model}-{config:02}-{index:03}'
                with (output/(stem+'.log')).open('xb') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                for name in ('report.json', 'serial.log', 'stage-monitor.txt', 'raw-returns.bin'):
                    (output/(stem+'-'+name)).write_bytes((case/name).read_bytes())
                completed += 1
                print(json.dumps(dict(completed=completed, total=plan['streams'], model=model,
                                      configuration=config, source=index)), flush=True)
    with (output/'complete.json').open('x') as receipt:
        receipt.write(json.dumps(dict(streams=completed, execution_sha256=sha(PLAN), retries=0))+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'run'))
    args = parser.parse_args()
    register() if args.action == 'register' else run()
