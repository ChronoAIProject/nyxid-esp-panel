"""Download, verify and extract a selected research snapshot into a fresh folder."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_archive(archive):
    with tarfile.open(archive) as bundle:
        names = set()
        for item in bundle.getmembers():
            path = PurePosixPath(item.name)
            if (path.is_absolute() or '..' in path.parts or not item.isfile()
                    or item.name in names):
                raise ValueError('Unsafe or duplicate archive member')
            names.add(item.name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-audio', action='store_true')
    p.add_argument('--include-history', action='store_true')
    p.add_argument('--asset', action='append',
                   help='Download only this named asset; repeat for incremental snapshots')
    args = p.parse_args()
    manifest = json.loads(Path(__file__).with_name('assets.json').read_text())
    if args.asset and set(args.asset) - {entry['name'] for entry in manifest['assets']}:
        p.error('Unknown requested asset')
    args.output.mkdir(parents=True, exist_ok=False)
    downloads = args.output / 'downloads'
    downloads.mkdir()
    for entry in manifest['assets']:
        name = entry['name']
        if args.asset and name not in args.asset:
            continue
        if name == 'synthetic-training-audio.tar.gz' and not (args.include_audio or args.asset):
            continue
        if name == 'historical-experiment-source.tar.gz' and not (args.include_history or args.asset):
            continue
        if Path(name).name != name:
            raise ValueError('Invalid asset name')
        path = downloads / name
        print('Downloading ' + name, flush=True)
        request = urllib.request.Request(manifest['base_url'] + '/' + name,
                                         headers={'User-Agent': 'nyx-wake-research'})
        with urllib.request.urlopen(request, timeout=120) as response, path.open('xb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Asset checksum mismatch: ' + name)
        verify_archive(path)
        with tarfile.open(path) as archive:
            archive.extractall(args.output, filter='data')
        contents = args.output / ('CONTENTS-' + name.removesuffix('.tar.gz') + '.json')
        for relative, expected in json.loads(contents.read_text()).items():
            value = args.output / relative
            if value.stat().st_size != expected['bytes'] or digest(value) != expected['sha256']:
                raise ValueError('Extracted file checksum mismatch: ' + relative)
        print('Verified ' + name, flush=True)
    print('Data root: ' + str(args.output.resolve() / manifest['data_root']))


if __name__ == '__main__':
    main()
