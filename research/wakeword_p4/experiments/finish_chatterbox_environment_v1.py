"""Finish isolated runtime only after base pip and source acquisition succeed."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path.cwd()
SOURCE = ROOT / '.local/wakeword/chatterbox-source-v1'
ENV = ROOT / '.local/wakeword/chatterbox-env-v1'
PERTH = 'ff1c8ac55a976971245cdd53c18d6131ca00d993'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert Path(sys.prefix).resolve() == ENV.resolve(), 'Use isolated venv python'
    assert shutil.disk_usage(ROOT).free > 100 * 1024**3
    assert importlib.metadata.version('torch') == '2.6.0'
    assert importlib.metadata.version('transformers') == '5.2.0'
    receipt = json.loads((SOURCE / 'result.json').read_text())
    for record in receipt['records']:
        path = ROOT / record['path']
        assert path.stat().st_size == record['bytes']
        assert sha(path) == record['sha256']
    runtime = SOURCE / 'runtime'
    runtime.mkdir(exist_ok=False)
    with tarfile.open(SOURCE / 'runtime.tar.gz') as archive:
        for member in archive.getmembers():
            assert member.isfile() or member.isdir(), 'No archive links'
            assert (runtime / member.name).resolve().is_relative_to(runtime.resolve())
        # Python 3.11 on this host predates extraction filters. The checks
        # above reject links, special files and paths outside the destination.
        archive.extractall(runtime)
    package, = runtime.iterdir()
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    f'git+https://github.com/resemble-ai/Perth.git@{PERTH}'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                    '--no-deps', str(package)], check=True)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    import torch
    torch.set_num_threads(2)
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    assert torch.cuda.is_available()
    assert '4090' in torch.cuda.get_device_name(0)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode()
    (SOURCE / 'environment-freeze.txt').write_text(freeze)
    result = dict(schema='chatterbox-environment-v1', perth_revision=PERTH,
                  runtime_archive_sha256=sha(SOURCE / 'runtime.tar.gz'),
                  freeze_sha256=sha(SOURCE / 'environment-freeze.txt'),
                  python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), turbo_import=True,
                  generation_started=False,
                  note='UI gradio omitted; runtime installed without dependency resolution; no model loaded yet.')
    (SOURCE / 'environment-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
