"""Retain the completed, rejected US continuation and original sampler replay.

This packages existing bytes; it does not train, score models or access audio.
Use --verify-public after uploading the two archives to the research prerelease.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive, SECRET
from fetch_assets import verify_archive


HERE = Path(__file__).resolve().parent
RELEASE = 'research-hey-nyx-20261110'
URL = 'https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/' + RELEASE + '/'


def verify_members(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        inventories = [n for n in bundle.getnames() if n.startswith('CONTENTS-')]
        assert len(inventories) == 1
        inventory = json.loads(bundle.extractfile(inventories[0]).read())
        assert set(bundle.getnames()) == set(inventory) | set(inventories)
        for name, expected in inventory.items():
            data = bundle.extractfile(name).read()
            assert len(data) == expected['bytes']
            assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
    return len(inventory)


def add_tree(archive, prefix, root):
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        assert not path.is_symlink()
        if path.suffix in ('.json', '.jsonl', '.log', '.py'):
            assert not SECRET.search(path.read_bytes()), path
        archive.add(prefix + '/' + path.relative_to(root).as_posix(), source=path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--continuation', type=Path)
    parser.add_argument('--sampler', type=Path)
    parser.add_argument('--verify-public', action='store_true')
    args = parser.parse_args()
    if args.verify_public:
        for path in sorted(args.output.glob('descriptor-*.json')):
            descriptor = json.loads(path.read_bytes())
            target = args.output / ('anonymous-' + descriptor['name'])
            request = urllib.request.Request(descriptor['url'], headers={'User-Agent': 'nyx-wake-research'})
            with urllib.request.urlopen(request, timeout=60) as response, target.open('xb') as output:
                while data := response.read(1024 * 1024):
                    output.write(data)
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            assert target.stat().st_size == descriptor['bytes'] and actual == descriptor['sha256']
            assert verify_members(target) == descriptor['listed_member_hashes_verified']
            descriptor.update(public_download_verified=True, anonymous_download_sha256=actual)
            path.write_text(json.dumps(descriptor, indent=2) + '\n')
            print(json.dumps(descriptor))
        return
    if any(p is None for p in (args.models, args.continuation, args.sampler)):
        parser.error('Build requires --models, --continuation and --sampler')
    args.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.continuation / 'result-summary.json').read_bytes())
    assert summary['accepted'] is False and summary['continuation_advanced'] is False
    original = json.loads((args.sampler / 'frozen-sampler-replay-v1/result.json').read_bytes())
    assert original['all_checks_pass'] is True and original['source_and_endpoint_draws'] == 1000
    for name in ('us-embedding-continuation-mac-v1', 'us-feature-sampler-mac-v1'):
        archive = Archive(args.output, name)
        if name.startswith('us-embedding'):
            add_tree(archive, 'run', args.continuation)
            for arm in ('control', 'treatment'):
                directory = 'head-us-source-' + arm + '-v1'
                add_tree(archive, 'models/' + directory, args.models / directory)
            add_tree(archive, 'features/head-features-us-single-v1', args.models / 'head-features-us-single-v1')
            for path in sorted((HERE / 'plans').glob('us-embedding-*-v1.json')):
                archive.add('plans/' + path.name, source=path)
            archive.add('source/train_head.py', source=HERE.parent / 'wakeword_p4/embedding/train_head.py')
            scope = 'Completed matched training/validation; both dense and equal-mean arms rejected. All eight checkpoints and score traces retained; not a deployable candidate.'
        else:
            add_tree(archive, 'replay', args.sampler / 'frozen-sampler-replay-v1')
            archive.add('pack-authentication-v1.json', source=args.sampler / 'pack-authentication-v1.json')
            scope = original['scope']
        archive.add('publication/package_us_mac_results.py', source=Path(__file__))
        descriptor = archive.close()
        descriptor.update(release_tag=RELEASE, accepted=False, scope=scope,
            listed_member_hashes_verified=verify_members(args.output / descriptor['name']),
            public_download_verified=False, source_audio_or_device_credentials_included=False,
            url=URL + descriptor['name'])
        (args.output / ('descriptor-' + name + '.json')).write_text(json.dumps(descriptor, indent=2) + '\n')
        print(json.dumps(descriptor))


if __name__ == '__main__':
    main()
