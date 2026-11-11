"""Publishable evidence for the frozen training-only source-sampling study.

Build locally, upload the resulting tarball to the research prerelease, then
run --verify-public to check a fresh anonymous download before committing its
manifest. No audio, complete feature cache or device state is packaged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive, SECRET
from fetch_assets import verify_archive

NAME = 'intent-source-balanced-sampling-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def members(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        name = 'CONTENTS-'+NAME+'.json'
        inventory = json.loads(bundle.extractfile(name).read())
        assert set(bundle.getnames()) == set(inventory)|{name}
        for member, expected in inventory.items():
            data = bundle.extractfile(member).read()
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], member
        return len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-public', action='store_true')
    for name in ('artifacts', 'failed-launch', 'preflight', 'initial-weights'):
        parser.add_argument('--'+name, type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    if args.verify_public:
        manifest = json.loads((output/'manifest.json').read_bytes())
        download = output/'unauthenticated-download.tar.gz'
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'nyx-wake-research'})
        with urllib.request.urlopen(request, timeout=60) as response, download.open('xb') as out:
            while data := response.read(1024*1024):
                out.write(data)
        assert sha(download) == manifest['sha256'] and download.stat().st_size == manifest['bytes']
        assert members(download) == manifest['listed_member_hashes_verified']
        manifest.update(public_download_verified=True, unauthenticated_download_sha256=sha(download))
    else:
        if any(getattr(args, n) is None for n in ('artifacts', 'failed_launch', 'preflight', 'initial_weights')):
            parser.error('Build requires --artifacts, --failed-launch, --preflight and --initial-weights')
        root = args.artifacts.resolve()
        read = lambda name: json.loads((root/name).read_bytes())
        plan, launch, result, audit = [read(n) for n in ('plan.json', 'launch.json', 'study/result.json', 'audit.json')]
        assert read('train-exit.json') == read('audit-exit.json') == {'returncode': 0}
        assert result == audit and result['accepted'] is False
        assert result['plan_sha256'] == launch['plan_sha256'] == sha(root/'plan.json')
        assert sha(args.initial_weights) == plan['initializer_sha256']
        assert json.loads((args.failed_launch/'train-exit.json').read_bytes()) == {'returncode': 1}
        assert 'Ran 8 tests' in args.preflight.read_text() and '\nOK\n' in args.preflight.read_text()
        report = {'scope': plan['scope'], 'training_source_commit': launch['checkout_revision'],
            'plan_sha256': sha(root/'plan.json'), 'launch_sha256': sha(root/'launch.json'),
            'raw_result_sha256': sha(root/'study/result.json'), 'audit_sha256': sha(root/'audit.json'),
            'sampling_inventory_sha256': plan['inventory_sha256'],
            'original_control_all_six_folds_replayed': True, 'final_fits': 12, 'software_tests': 8,
            'software_log_sha256': sha(args.preflight), 'diagnostic_gate': plan['diagnostic_gate'],
            'engine_macro_accuracy': result['engine_macro_accuracy'],
            'aggregate_held_engine': result['aggregate_held_engine'],
            'full_fit': {arm: result['folds']['full-fit'][arm]['summary'] for arm in ('control', 'balanced')},
            'held_engine_macro_by_fold': {f: {a: values[a]['summary']['macro_accuracy'] for a in values}
                for f, values in result['folds'].items() if f != 'full-fit'},
            'advance': result['advance'], 'accepted': False,
            'first_launch': 'Wrong bank directory; failed before loading samples or fitting. Original failed launch and logs retained; corrected-path launch uses identical frozen source/plan.',
            'audit_scope': 'Training cache/index/bank bytes, fold/source coverage, all probability vectors by independent rational counts, draw schedules, saved-score classifications, exact control final tensors/scores/decodes and gate. Does not independently replay treatment optimization or establish wake quality.',
            'limits': plan['limits']}
        (root/'result-summary.json').write_text(json.dumps(report, indent=2)+'\n')
        output.mkdir(parents=True, exist_ok=False); archive = Archive(output, NAME)
        for prefix, source in (('completed', root), ('first-launch-failed', args.failed_launch.resolve())):
            for path in sorted(source.rglob('*')):
                if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                    continue
                assert not path.is_symlink() and path.suffix in ('.json', '.jsonl', '.log', '.py', '.h5', '.npy')
                if path.suffix in ('.json', '.jsonl', '.log', '.py'):
                    assert not SECRET.search(path.read_bytes()), path
                archive.add(prefix+'/'+path.relative_to(source).as_posix(), source=path)
        archive.add('preflight/tests.log', source=args.preflight)
        archive.add('inputs/initial.weights.h5', source=args.initial_weights)
        archive.add('publication/package_intent_sampling.py', source=Path(__file__))
        manifest = archive.close(); count = members(output/manifest['name'])
        manifest.update(release_tag='research-hey-nyx-20261110',
            training_source_commit=launch['checkout_revision'], diagnostic_advance=result['advance'], accepted=False,
            dependencies=['compact-broad-bank-v1.tar.gz', 'intent-phone-generalization-v1.tar.gz'],
            all_final_checkpoints_included=True, complete_intent_scores_included=True,
            full_scalar_feature_cache_included=False, failed_initial_launch_included=True,
            listed_member_hashes_verified=count, public_download_verified=False,
            url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+manifest['name'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
