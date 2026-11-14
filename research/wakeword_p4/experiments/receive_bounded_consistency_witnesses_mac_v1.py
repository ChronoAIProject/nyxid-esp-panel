"""Authenticate original bounded-family diagnostic bytes; no inference."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile


GPU_REF = 'ee0883fd15d058cb6ddd3040e41880bedfe70838'
PLAN_REF = 'db77c3efd8ad8bbbd5612df189080a2c539d8d39'
ROOT = 'research/wakeword_p4/'
RESULTS = ROOT + 'results/'
MANIFEST = RESULTS + '4090-bounded-consistency-fixed-grid-mac-handoff-manifest-v1.json'
RECEIPT = RESULTS + '4090-bounded-consistency-fixed-grid-mac-handoff-v1.json'
PLAN = RESULTS + 'mac-bounded-consistency-fixed-grid-witness-replay-plan-v1.json'
ARCHIVE_SHA = 'f4a3933ae0d736386eaa74c9db29273bf7a91cc779cc5f1f02670267e0ec0343'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tracked(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path])


def safe_name(name):
    path = PurePosixPath(name)
    assert name and not path.is_absolute() and '..' not in path.parts
    assert str(path) == name


def receive(a):
    receipt = json.loads(tracked(GPU_REF, RECEIPT))
    archive = a.archive.read_bytes()
    assert len(archive) == receipt['bytes'] == 1102004
    assert digest(archive) == receipt['sha256'] == ARCHIVE_SHA
    with tarfile.open(a.archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == receipt['exact_entries'] == 32
        assert len({m.name for m in members}) == len(members)
        for member in members:
            safe_name(member.name)
            assert member.isfile() and 0 <= member.size < 1000000
        content = {m.name: tar.extractfile(m).read() for m in members}
    manifest_bytes = content['manifest.json']
    assert manifest_bytes == tracked(GPU_REF, MANIFEST)
    assert digest(manifest_bytes) == receipt['manifest_sha256']
    manifest = json.loads(manifest_bytes)
    assert manifest['fixed_requests'] == dict(hey_willow=14, hey_nyx=48394)
    files = manifest['files']
    assert len(files) == receipt['payloads'] == 31
    assert {f['name'] for f in files} == set(content) - {'manifest.json'}
    verified = {}
    for item in files:
        name = item['name']
        safe_name(name)
        assert len(content[name]) == item['bytes'] and digest(content[name]) == item['sha256']
        path = item['source_receipt']
        safe_name(path)
        original = tracked(item['source_receipt_commit'], path)
        assert digest(original) == item['source_receipt_sha256']
        verified[(item['source_receipt_commit'], path)] = digest(original)
        if name.startswith(('receipts/', 'source/')):
            assert content[name] == original
    plan_bytes = tracked(PLAN_REF, PLAN)
    assert Path(PLAN).read_bytes() == plan_bytes
    plan = json.loads(plan_bytes)
    by_source = {f['source_path']: f for f in files}
    assert len(by_source) == len(files)
    published = lambda name: json.loads(content['receipts/4090-hey-broad-bounded-consistency-' + name + '-v1.json'])
    formal, screen = published('formal-result'), published('fixed-grid-screen-result')
    numerical = published('fixed-grid-screen-independent-audit')
    assert formal['all_passed'] and len(formal['cases']) == len(plan['arms']) == 4
    assert len(screen['cases']) == 540
    assert numerical['qualified_complete270_conditions'] == []
    assert numerical['result_sha256'] == digest(content['receipts/4090-hey-broad-bounded-consistency-fixed-grid-screen-result-v1.json'])
    expected = {(target, condition) for target in ['hey_willow', 'hey_nyx']
                for condition in ['raw_logit_control', 'bounded_logit_consistency']}
    assert {(r['target'], r['condition']) for r in plan['arms']} == expected
    for arm in plan['arms']:
        target, condition = arm['target'], arm['condition']
        for path_key, sha_key in [('model_path', 'model_sha256'), ('weights_path', 'weights_sha256')]:
            item = by_source[arm[path_key]]
            assert item['sha256'] == arm[sha_key]
            if path_key == 'model_path':
                assert item['bytes'] == arm['model_bytes']
        run, = [r for r in formal['cases'] if (r['target'], r['condition']) == (target, condition)]
        final = run['final_repeat']
        assert run['steps'] == final['steps'] == 10000
        assert final['weights_sha256'] == arm['weights_sha256']
        training = str(PurePosixPath(arm['weights_path']).parent / 'training.json')
        assert json.loads(content[by_source[training]['name']]) == final
        order = next(c['request_order'] for c in plan['inputs'] if c['target'] == target)
        row, = [r for r in screen['cases'] if
                (r['target'], r['condition'], r['request_order']) == (target, condition, order)]
        item = by_source[row['arrays_path']]
        assert item['sha256'] == row['arrays_sha256']
        assert item['name'] == f'witnesses/{target}-{condition}-request{order}.npz'
    for case in plan['inputs']:
        assert digest((a.old_feature_payload / case['path']).read_bytes()) == case['arrays_sha256']
    assert not a.report.exists()
    a.output_dir.mkdir(exist_ok=False)
    for name, data in content.items():
        path = a.output_dir / 'archive-members' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
    copied = 0
    for item in files:
        source = item['source_path']
        if source.startswith('.local/wakeword/'):
            safe_name(source)
            path = a.output_dir / 'replay-payload' / source
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(content[item['name']])
            copied += 1
    report = dict(schema='mac-bounded-consistency-original-handoff-intake-v1',
        gpu_ref=GPU_REF, published_mac_plan_ref=PLAN_REF,
        script_sha256=digest(Path(__file__).read_bytes()), archive_bytes=len(archive),
        archive_sha256=digest(archive), manifest_sha256=digest(manifest_bytes),
        plan_sha256=digest(plan_bytes), payload_files=len(files),
        source_receipts_verified=len(verified), original_relative_copies=copied,
        actual_four_model_h5_locks_exact=True, original_training_receipts_match=True,
        four_fixed_screen_npz_locks_exact=True, original_feature_npz_locks_exact=True,
        members=[dict(name=f['name'], bytes=f['bytes'], sha256=f['sha256']) for f in files],
        all_passed=True,
        limits='Actual transport/identity verification only. Complete540 rejects both conditions; preserve all failures. No model invoke, fresh development, independent acceptance, recognition/FAR, P4 or candidate qualification. Native replay remains a separate diagnostic.')
    with a.report.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'members'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--old-feature-payload', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    receive(parser.parse_args())
