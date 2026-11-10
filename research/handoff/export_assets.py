"""Export selected public research artifacts without build caches or device state.

Feature values, model weights and original audit metadata remain byte-exact.
Only absolute feature-index paths are made relative; portable training plans
explicitly bind the new index hashes and retain their original identities.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()


SECRET = re.compile(rb'sk-(?:proj-)?[A-Za-z0-9_-]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


class Archive:
    def __init__(self, output, name):
        self.path = output / (name + '.tar.gz')
        self.tar = tarfile.open(self.path, 'w:gz', compresslevel=3, dereference=True)
        self.files = {}

    def add(self, name, source=None, data=None):
        if name in self.files:
            return
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe archive path')
        data = source.read_bytes() if data is None else data
        if path.suffix in ('.json', '.jsonl', '.py', '.h', '.c', '.cpp', '.txt', '.md') and SECRET.search(data):
            raise ValueError('Possible secret in ' + name)
        item = tarfile.TarInfo(name)
        item.size = len(data)
        item.mode = 0o644
        self.tar.addfile(item, io.BytesIO(data))
        self.files[name] = {'sha256': sha(data), 'bytes': len(data)}

    def close(self):
        self.add('CONTENTS-' + self.path.name.removesuffix('.tar.gz') + '.json',
                 data=encoded(self.files))
        self.tar.close()
        return {'name': self.path.name, 'sha256': sha(self.path.read_bytes()),
                'bytes': self.path.stat().st_size, 'files': len(self.files),
                'uncompressed_bytes': sum(f['bytes'] for f in self.files.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-output', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.source_output.resolve()
    base = root / 'hey-nyx-training-v3/embedding-feasibility'
    repo = Path(__file__).resolve().parents[2]
    prefix = 'workspace/hey-nyx-training-v3/embedding-feasibility/'
    assets = []
    features = Archive(args.output, 'embedding-features')
    rebound, sources = {}, {}
    for index in sorted(base.glob('head-features-*/index.jsonl')):
        name = index.parent.name
        if name == 'head-features-pilot':
            continue
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        if any(r['split'] not in ('training', 'validation') for r in rows):
            raise ValueError('Unexpected held-out features')
        for row in rows:
            old = Path(row['path'])
            source = old if old.is_absolute() else base / old
            relative = source.relative_to(base)
            data = source.read_bytes()
            if sha(data) != row['features_sha256']:
                raise ValueError('Feature checksum mismatch')
            row['path'] = str(relative)
            features.add(prefix + str(relative), data=data)
        normalized = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()
        # Keep original bytes when already portable, preserving original hashes.
        original = index.read_bytes()
        if all(not Path(json.loads(l)['path']).is_absolute() for l in original.splitlines()):
            normalized = original
        features.add(prefix + name + '/index.jsonl', data=normalized)
        if normalized != original:
            features.add(prefix + name + '/original-index.jsonl', data=original)
        for file in ('identity.json', 'complete.json', 'jobs-private.json'):
            features.add(prefix + name + '/' + file, source=index.parent / file)
        jobs = json.loads((index.parent / 'jobs-private.json').read_text())
        for job in jobs:
            if job['kind'] == 'tts':
                sources[job['source']] = job['source_sha256']
        rebound[name] = {'original_index_sha256': sha(original), 'index_sha256': sha(normalized),
                         'complete_sha256': sha((index.parent / 'complete.json').read_bytes()),
                         'rows': len(rows)}
    for plan in sorted((repo / 'research/wakeword_p4/embedding').glob('*plan*.json')):
        value = json.loads(plan.read_text())
        changed = False
        for item in value.get('extra_training_features', []):
            if item['directory'] in rebound:
                binding = rebound[item['directory']]
                if item['index_sha256'] != binding['original_index_sha256']:
                    raise ValueError('Plan does not bind original feature index')
                item['index_sha256'] = binding['index_sha256']
                item['complete_sha256'] = binding['complete_sha256']
                changed = changed or item['index_sha256'] != binding['original_index_sha256']
        if changed:
            value['handoff_provenance'] = {'original_plan_sha256': sha(plan.read_bytes()),
                'change': 'Only feature-index path bindings; PCM/arrays/model/recipe/splits unchanged'}
        features.add(prefix + 'portable-plans/' + plan.name,
                     data=encoded(value) if changed else plan.read_bytes())
    features.add(prefix + 'portable-feature-provenance.json', data=encoded(rebound))
    assets.append(features.close())
    print(json.dumps(assets[-1]), flush=True)

    models = Archive(args.output, 'models-and-checkpoints')
    allowed = {'.tflite', '.keras', '.h5', '.json'}
    for directory in sorted(base.glob('head-*')):
        if not directory.is_dir() or directory.name.startswith(('head-features-', 'head-negative-')):
            continue
        for source in sorted(directory.rglob('*')):
            if source.is_file() and (source.suffix in allowed or source.name == 'representatives-training-only.npy'):
                models.add(prefix + str(source.relative_to(base)), source=source)
    for source in sorted(base.iterdir()):
        if source.is_file() and (source.suffix in ('.tflite', '.npy') or
                source.name.startswith(('head-', 'stream-', 'native-mel')) and source.suffix == '.json'):
            models.add(prefix + source.name, source=source)
    for directory in sorted(base.glob('runtime-equal-mean-bundle-v*')):
        for source in sorted(directory.iterdir()):
            if source.is_file():
                models.add(prefix + str(source.relative_to(base)), source=source)
    for version in (1, 2, 3):
        for directory in sorted((root / f'hey-nyx-training-v{version}').glob('model*')):
            if not directory.is_dir():
                continue
            for source in sorted(directory.rglob('*')):
                if source.is_file() and source.suffix in allowed and 'source' not in source.relative_to(directory).parts:
                    models.add('workspace/' + str(source.relative_to(root)), source=source)
    assets.append(models.close())
    print(json.dumps(assets[-1]), flush=True)

    arrays = Archive(args.output, 'evaluation-arrays')
    for directory in sorted(base.glob('head-*')):
        if not directory.is_dir() or directory.name.startswith(('head-features-', 'head-negative-')):
            continue
        for source in sorted(directory.rglob('*.npy')):
            if source.name != 'representatives-training-only.npy':
                arrays.add(prefix + str(source.relative_to(base)), source=source)
    assets.append(arrays.close())
    print(json.dumps(assets[-1]), flush=True)

    audio = Archive(args.output, 'synthetic-training-audio')
    for name, expected in sorted(sources.items()):
        source = Path(name)
        data = source.read_bytes()
        if sha(data) != expected:
            raise ValueError('Synthetic source checksum mismatch')
        audio.add('workspace/' + str(source.relative_to(root)), data=data)
    feature_root = root / 'hey-nyx-training-v3/features-expanded-human335'
    for name in ('features.json', 'generation.json', 'human.json', 'ambient.json'):
        audio.add('workspace/' + str((feature_root / name).relative_to(root)), source=feature_root / name)
    assets.append(audio.close())
    print(json.dumps(assets[-1]), flush=True)

    code = Archive(args.output, 'historical-experiment-source')
    exclude = {'build', 'managed_components', '.git', '__pycache__'}
    for version in (1, 2, 3):
        parent = root / f'hey-nyx-training-v{version}'
        for source in sorted(parent.rglob('*')):
            relative = source.relative_to(parent)
            if any(part in exclude or part.startswith('build-') for part in relative.parts):
                continue
            if source.is_file() and (source.suffix == '.py' or
                    source.suffix in ('.cpp', '.c', '.h', '.cmake', '.lf') and
                    'embedding-feasibility' in relative.parts):
                # Exclude private audio-capture headers generated for one LAN.
                if 'private' in source.name or source.name in ('devices_data.h', 'secrets.h'):
                    continue
                code.add('historical-source/' + str(source.relative_to(root)), source=source)
    assets.append(code.close())
    print(json.dumps(assets[-1]), flush=True)
    manifest = {'format': 'nyx-wake-research-assets-v1', 'assets': assets,
        'scope': 'Research only; no device credentials, private microphone recordings, firmware/NVS backups or build caches',
        'features': rebound, 'synthetic_sources': len(sources),
        'data_root': 'workspace/hey-nyx-training-v3/embedding-feasibility',
        'excluded': ['private room/microphone captures', 'board state, tokens, NVS and firmware backups',
                     'build and dependency caches', 'large redownloadable public raw negative corpora',
                     'untouched final testing corpora', 'redundant old derived microWakeWord features'],
        'provenance': 'Source paths in original jobs are audit/corpus metadata, not required for head training. The jobs-private filename is historical; these exported rows describe only synthetic/public datasets.'}
    (args.output / 'assets.json').write_bytes(encoded(manifest))


if __name__ == '__main__':
    main()
