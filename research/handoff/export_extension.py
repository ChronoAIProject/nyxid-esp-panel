"""Export one completed training-feature/model extension for an existing snapshot.

No input microphone recordings, device state or build directories are read.
Feature arrays/checkpoints stay exact; only portable plan/index bindings change.
"""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--training', action='append', required=True)
    p.add_argument('--plan', type=Path, action='append', required=True)
    p.add_argument('--evaluation', action='append', default=[],
                   help='Explicit completed equal-mean evaluation directory')
    p.add_argument('--evaluation-plan', type=Path, action='append', default=[])
    p.add_argument('--models-only', action='store_true',
                   help='Reuse an already published feature extension without duplicating its arrays')
    p.add_argument('--name', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if len(args.training) != len(args.plan):
        p.error('Each training directory requires one corresponding plan')
    if len(args.evaluation) != len(args.evaluation_plan):
        p.error('Each evaluation directory requires one corresponding plan')
    for name in (args.features, *args.training, *args.evaluation, args.name):
        if Path(name).parts != (name,) or name in ('.', '..'):
            p.error('Names must be single path components')
    manifest = json.loads(Path(__file__).with_name('assets.json').read_text())
    base = args.data_root.resolve()
    features = base / args.features
    complete = json.loads((features / 'complete.json').read_text())
    if not complete['all_training_only'] or complete.get('pilot_only', False):
        raise ValueError('Only completed full training views may be exported')
    original = (features / 'index.jsonl').read_bytes()
    rows = list(map(json.loads, original.splitlines()))
    if len(rows) != complete['jobs'] or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Feature completion or unique identity differs')
    for directory, plan_path in zip(args.training, args.plan):
        fitted = json.loads((base / directory / 'complete.json').read_text())
        if fitted['plan_sha256'] != sha(plan_path.read_bytes()):
            raise ValueError('Training does not bind the explicit plan')
    for directory, plan_path in zip(args.evaluation, args.evaluation_plan):
        calibrated = json.loads((base / directory / 'calibration.json').read_text())
        if calibrated['plan_sha256'] != sha(plan_path.read_bytes()):
            raise ValueError('Evaluation does not bind the explicit plan')
    args.output.mkdir(parents=True, exist_ok=False)
    archive = Archive(args.output, args.name)
    prefix = manifest['data_root'].rstrip('/') + '/'
    for row in rows:
        if row['split'] != 'training':
            raise ValueError('Extension cannot add held-out rows')
        path = Path(row['path'])
        path = path if path.is_absolute() else base / path
        relative = path.resolve().relative_to(base)
        data = path.read_bytes()
        if sha(data) != row['features_sha256']:
            raise ValueError('Feature array identity differs')
        if not args.models_only:
            archive.add(prefix + str(relative), data=data)
        row['path'] = str(relative)
    normalized = ('\n'.join(json.dumps(r) for r in rows) + '\n').encode()
    if not args.models_only:
        archive.add(prefix + args.features + '/index.jsonl', data=normalized)
        archive.add(prefix + args.features + '/original-index.jsonl', data=original)
        for name in ('identity.json', 'complete.json', 'jobs-private.json'):
            archive.add(prefix + args.features + '/' + name, source=features / name)
    binding = {'original_index_sha256': sha(original), 'index_sha256': sha(normalized),
        'complete_sha256': sha((features / 'complete.json').read_bytes()), 'rows': len(rows)}
    bindings = {**manifest['features'], args.features: binding}
    for directory, plan_path in zip(args.training, args.plan):
        plan = json.loads(plan_path.read_text())
        for entry in plan.get('extra_training_features', []):
            source = bindings[entry['directory']]
            if (entry['index_sha256'] != source['original_index_sha256']
                    or entry['complete_sha256'] != source['complete_sha256']):
                raise ValueError('Original plan feature provenance differs')
            entry['index_sha256'] = source['index_sha256']
        plan['handoff_provenance'] = {'original_plan_sha256': sha(plan_path.read_bytes()),
            'change': 'Only feature-index path bindings; arrays,recipe,splits and model weights unchanged'}
        archive.add(prefix + 'portable-plans/' + plan_path.name, data=encoded(plan))
    for plan_path in args.evaluation_plan:
        archive.add(prefix + 'portable-plans/' + plan_path.name, source=plan_path)
    for directory in [*args.training, *args.evaluation]:
        for path in sorted((base / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.keras', '.tflite', '.json', '.npy'):
                path.resolve().relative_to(base)
                archive.add(prefix + str(path.relative_to(base)), source=path)
    entry = archive.close()
    result = {'asset': entry, 'feature_directory': args.features, 'feature_binding': binding,
        'portable_plans': ['portable-plans/' + f.name for f in [*args.plan, *args.evaluation_plan]],
        'includes_feature_arrays': not args.models_only}
    (args.output / 'extension.json').write_bytes(encoded(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
