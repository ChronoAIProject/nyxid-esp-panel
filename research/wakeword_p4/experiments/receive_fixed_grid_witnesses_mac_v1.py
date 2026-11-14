"""Authenticate the published original fixed-grid handoff; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


GPU_REF = '201bd0f39a315aa5be89abf464c319f1aa780118'
ARCHIVE_SHA = '27677e306b34471b82d4dd6eb367192e7e389c55ec201f36a887f93cff891d8b'
PREFIX = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tracked(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path])


def safe_name(name):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and '..' not in path.parts
    assert str(path) == name
    return path


def run(a):
    archive = a.archive.read_bytes()
    assert len(archive) == 839771 and digest(archive) == ARCHIVE_SHA
    with tarfile.open(a.archive, 'r:gz') as tar:
        entries = tar.getmembers()
        assert len(entries) == 27
        assert len({m.name for m in entries}) == len(entries)
        for member in entries:
            safe_name(member.name)
            assert member.isfile() and 0 <= member.size < 1000000
        content = {m.name: tar.extractfile(m).read() for m in entries}
    manifest_bytes = content['manifest.json']
    assert manifest_bytes == tracked(
        GPU_REF, PREFIX + 'results/4090-fixed-grid-mac-handoff-manifest-v1.json')
    manifest = json.loads(manifest_bytes)
    files = manifest['files']
    assert len(files) == 26
    assert {f['name'] for f in files} == set(content) - {'manifest.json'}
    receipts = {}
    for item in files:
        name = item['name']
        safe_name(name)
        data = content[name]
        assert len(data) == item['bytes'] and digest(data) == item['sha256']
        path = item['source_receipt']
        if path == '/tmp/mac-fixed-grid-witness-replay-plan-v1.json':
            path = PREFIX + 'results/mac-fixed-grid-witness-replay-plan-v1.json'
        safe_name(path)
        receipt = tracked(item['source_receipt_commit'], path)
        assert digest(receipt) == item['source_receipt_sha256']
        receipts[(item['source_receipt_commit'], path)] = digest(receipt)
        if name.startswith('metadata/'):
            assert data == receipt
    plan_bytes = tracked('f4685c5', PREFIX + 'results/mac-fixed-grid-witness-replay-plan-v1.json')
    assert content['metadata/mac-fixed-grid-witness-replay-plan-v1.json'] == plan_bytes
    plan = json.loads(plan_bytes)
    by_source = {f['source_path']: f for f in files}
    for arm in plan['arms']:
        for path_key, sha_key in [('model_path', 'model_sha256'), ('weights_path', 'weights_sha256')]:
            item = by_source[arm[path_key]]
            assert item['sha256'] == arm[sha_key]
            if path_key == 'model_path':
                assert item['bytes'] == arm['model_bytes']
    for item in plan['inputs']:
        received = content[by_source[item['path']]['name']]
        original = (a.old_feature_payload / item['path']).read_bytes()
        assert digest(original) == item['arrays_sha256'] and received == original
    # Materialize copies under their original relative names for the unchanged
    # frozen replay entry point. Published receipts identify private file
    # provenance; they do not assert those private blobs were tracked by Git.
    a.output_dir.mkdir(exist_ok=False)
    for name, data in content.items():
        dest = a.output_dir / 'archive-members' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(data)
    for item in files:
        source_path = item['source_path']
        if not source_path.startswith('.local/wakeword/'):
            continue
        safe_name(source_path)
        dest = a.output_dir / 'replay-payload' / source_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as f:
            f.write(content[item['name']])
    report = dict(
        schema='mac-fixed-grid-original-handoff-intake-v1', gpu_ref=GPU_REF,
        script_sha256=digest(Path(__file__).read_bytes()),
        archive_bytes=len(archive), archive_sha256=digest(archive),
        manifest_sha256=digest(manifest_bytes), plan_sha256=digest(plan_bytes),
        payload_files=26, original_relative_copies=sum(
            f['source_path'].startswith('.local/wakeword/') for f in files),
        source_receipts_verified=len(receipts), four_model_h5_locks_exact=True,
        two_original_feature_npz_byte_exact=True,
        members=[dict(name=f['name'], bytes=f['bytes'], sha256=f['sha256']) for f in files],
        all_passed=True,
        limits='Transport and published identity verification only. No inference, new audio IO, recognition, model admission, fresh development, acceptance or board operation. Four candidates remain rejected.')
    with a.report.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'members'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--old-feature-payload', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    run(parser.parse_args())
