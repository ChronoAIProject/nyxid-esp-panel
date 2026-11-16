"""Fixed CUDA small.en transcript screen, no wake scores or relabeling."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('research/wakeword_p4')
DATA = Path('.local/wakeword/chatterbox-conditioned-training-v1')
MODEL = Path('.local/wakeword/whisper-small-en')
MODEL_SOURCE = Path('.local/wakeword/whisper-small-en-source.json')
RUNNER = ROOT / 'training/audit_comparison_asr.py'
PLAN = ROOT / 'results/4090-chatterbox-training-asr-plan-v1.json'
OUT = Path('.local/wakeword/chatterbox-training-asr-v1')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lock():
    complete = DATA / 'completed.json'
    assert json.loads(complete.read_text())['requests'] == 96
    rows = [json.loads((DATA / f'{i:03d}.json').read_text()) for i in range(96)]
    paths = [Path(__file__), RUNNER, MODEL_SOURCE, complete]
    paths += [p for p in MODEL.iterdir() if p.is_file()]
    manifest = []
    for i, row in enumerate(rows):
        assert row['index'] == i and row['request']['role'] == 'training-only'
        paths.append(DATA / f'{i:03d}.json')
        for item in row['files']:
            path = Path(item['path'])
            assert sha(path) == item['sha256']
            paths.append(path)
        wav, = [item for item in row['files'] if item['path'].endswith('.wav')]
        request = row['request']
        manifest.append(dict(path=Path(wav['path']).name, split='training',
            speakers=[request['voice']], label=request['requested_label'],
            text=request['text'], sha256=wav['sha256']))
    manifest_path = DATA / 'manifest.jsonl'
    with manifest_path.open('x') as stream:
        stream.write(''.join(json.dumps(row) + '\n' for row in manifest))
    paths.append(manifest_path)
    plan = dict(files={str(path): sha(path) for path in paths},
        versions={name: importlib.metadata.version(name) for name in
                  ['faster-whisper', 'ctranslate2', 'numpy', 'soundfile']},
        policy='All96 training-only clips, fixed existing small.en CUDA float16 beam3 no prompt/VAD. Requested label retained; transcript consistency is not acoustic truth or phone alignment. No filtering, wake scoring or development/acceptance access.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(plan, indent=2) + '\n')


def run():
    assert subprocess.check_output(['git', 'show',
        'origin/research-hye-nyx-voice-gpu:' + str(PLAN)]) == PLAN.read_bytes()
    plan = json.loads(PLAN.read_text())
    for path, expected in plan['files'].items():
        assert sha(Path(path)) == expected, path
    for name, expected in plan['versions'].items():
        assert importlib.metadata.version(name) == expected
    OUT.mkdir(exist_ok=False)
    subprocess.run([sys.executable, str(RUNNER), '--dataset', str(DATA),
        '--model', str(MODEL), '--device', 'cuda', '--model-description',
        'Pinned existing small.en CUDA float16 beam3 no prompt/VAD',
        '--model-source', str(MODEL_SOURCE), '--training-source',
        '--details', str(OUT / 'details.jsonl'), '--output', str(OUT / 'result.json')], check=True)


if __name__ == '__main__':
    {'lock': lock, 'run': run}[sys.argv[1]]()
