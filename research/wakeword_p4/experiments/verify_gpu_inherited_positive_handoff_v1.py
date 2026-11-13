"""Authenticate the published inherited-positive captures and rerun Mac comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ARCHIVE_SIZE = 17689908
ARCHIVE_SHA = '7d6eb2adc84fad0ebfc2889fa25874a859d835cad0101acc2e07691d2cd9bbd3'
GPU_COMMIT = 'a280268'
BASE_COMMIT = '43254c0'
TOOLS_COMMIT = '280c37728cc1823caecb19c4569e745077a0f372'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def git_file(commit, name):
    return subprocess.check_output(['git', 'show', commit + ':' + name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'extracted', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.extracted.exists() and not args.output.exists()
    assert args.archive.stat().st_size == ARCHIVE_SIZE
    assert sha(args.archive.read_bytes()) == ARCHIVE_SHA
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)) == 459
        assert all(m.isfile() for m in members)
        for name in names:
            path = Path(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert path.as_posix() == name
        manifest_blob = archive.extractfile('manifest.json').read()
        assert manifest_blob == git_file(GPU_COMMIT,
            'research/wakeword_p4/results/4090-inherited-positive-cpu-probe-manifest-v1.json')
        manifest = json.loads(manifest_blob)
        entries = manifest['files']
        assert len(entries) == 458
        assert len({e['path'] for e in entries}) == len(entries)
        assert set(names) == {e['path'] for e in entries} | {'manifest.json'}
        for entry in entries:
            blob = archive.extractfile(entry['path']).read()
            assert len(blob) == entry['size'] and sha(blob) == entry['sha256']
        # Validate the full archive before creating any extracted payload.
        args.extracted.mkdir(parents=True)
        for member in members:
            destination = args.extracted / member.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.extractfile(member).read())
    root = args.extracted
    tools = Path(__file__).resolve().parent
    origin = json.loads((root / 'tools/source-receipt.json').read_text())
    assert origin['source_commit'] == TOOLS_COMMIT
    for entry in origin['files']:
        payload = (root / 'tools' / Path(entry['path']).name).read_bytes()
        assert sha(payload) == entry['sha256']
        assert payload == git_file(TOOLS_COMMIT, entry['path'])
        if entry['path'].endswith('.py') or entry['path'].endswith('.patch'):
            assert payload == (tools / Path(entry['path']).name).read_bytes()
    source_paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
        BASE_COMMIT, 'research/wakeword_p4/training'], text=True).splitlines()
    source_paths = [name for name in source_paths if name.endswith('.py')]
    trainer = 'research/wakeword_p4/training/train_stream.py'
    for directory in ('base-source', 'patched-source'):
        actual = {p.relative_to(root / directory).as_posix()
                  for p in (root / directory).rglob('*.py')}
        assert actual == set(source_paths)
        for name in source_paths:
            blob = (root / directory / name).read_bytes()
            if directory == 'patched-source' and name == trainer:
                assert sha(blob) == 'b97b2debc92ca106ab199af60ac144aa7a78ce4a26e39621bc5b179fb637917e'
            else:
                assert blob == git_file(BASE_COMMIT, name)
    comparisons = []
    comparator = tools / 'compare_inherited_positive_integration_v1.py'
    for name in ('complete-off', 'complete-on'):
        case = root / name
        derived = case / 'mac-independent-comparison.json'
        subprocess.run([sys.executable, str(comparator), '--original', str(case / 'original'),
            '--disabled', str(case / 'disabled'), '--enabled', str(case / 'enabled'),
            '--output', str(derived)], check=True, capture_output=True, text=True)
        assert derived.read_bytes() == (case / 'comparison.json').read_bytes()
        committed = 'research/wakeword_p4/results/4090-inherited-positive-' + name + '-comparison-v1.json'
        assert derived.read_bytes() == git_file(GPU_COMMIT, committed)
        comparisons.append(dict(case=name, comparison_sha256=sha(derived.read_bytes()),
            comparison_byte_exact=True, comparison=json.loads(derived.read_text())))
    report = dict(schema='mac-gpu-inherited-positive-independent-receipt-v1',
        source_sha256=sha(Path(__file__).read_bytes()),
        comparator_sha256=sha(comparator.read_bytes()),
        archive_bytes=ARCHIVE_SIZE, archive_sha256=ARCHIVE_SHA,
        manifest_sha256=sha(manifest_blob), payload_count=len(entries),
        all_payload_hashes_exact=True, original_tools_and_source_snapshots_exact=True,
        captures=comparisons, training_reexecuted=False, acceptance_claim=False,
        limits='Anonymous archive integrity, authenticated sources and independent comparator replay only. '
               'Mac did not run original missing arrays or the optimizer. Twelve captured updates '
               'establish input integration, not wake quality, historical command counts or root cause. '
               'The restored eight rows change mixed source distribution; no detector promoted.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(payloads_verified=len(entries), all_comparisons_byte_exact=True)))


if __name__ == '__main__':
    main()
