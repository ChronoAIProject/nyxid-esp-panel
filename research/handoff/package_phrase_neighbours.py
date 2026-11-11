"""Archive training-only neighbour evidence, then verify its public download."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive, SECRET
from fetch_assets import verify_archive

NAME = 'phrase-neighbours-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_members(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        inventory_name = 'CONTENTS-'+NAME+'.json'
        inventory = json.loads(bundle.extractfile(inventory_name).read())
        assert set(bundle.getnames()) == set(inventory)|{inventory_name}
        for name, expected in inventory.items():
            data = bundle.extractfile(name).read()
            assert len(data) == expected['bytes'] and hashlib.sha256(data).hexdigest() == expected['sha256'], name
        return len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-public', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.verify_public:
        manifest = json.loads((output/'manifest.json').read_bytes())
        download = output/'unauthenticated-download.tar.gz'
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'nyx-wake-research'})
        with urllib.request.urlopen(request, timeout=60) as response, download.open('xb') as out:
            while block := response.read(1024*1024):
                out.write(block)
        assert sha(download) == manifest['sha256'] and download.stat().st_size == manifest['bytes']
        assert verify_members(download) == manifest['listed_member_hashes_verified']
        manifest.update(public_download_verified=True, unauthenticated_download_sha256=sha(download))
    else:
        if args.artifacts is None:
            parser.error('--artifacts is required to build')
        root = args.artifacts.resolve(); read = lambda name: json.loads((root/name).read_bytes())
        result, launch = read('result.json'), read('launch.json')
        audit_path = root.with_name(root.name+'-audit.json')
        audit = json.loads(audit_path.read_bytes())
        assert audit['result_sha256'] == sha(root/'result.json') and result['accepted'] is False
        assert audit['checked_pair_distances'] == result['computed_pair_distances']
        assert audit['all_features_match_original_cache'] and audit['all_comparisons_cohorts_and_review_selection_reconstructed']
        assert audit['maximum_independent_distance_delta'] <= audit['numeric_allowance']
        auditor = Path(__file__).with_name('verify_phrase_neighbours.py')
        assert sha(auditor) == audit['auditor_sha256']
        tests = root.with_name('phrase-neighbours-preflight.log')
        assert 'Ran 3 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
        output.mkdir(parents=True, exist_ok=False); archive = Archive(output, NAME)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            assert not path.is_symlink() and path.suffix in ('.py', '.json', '.npz')
            archive.add('study/'+path.relative_to(root).as_posix(), source=path)
        extras = {'audit/audit.json': audit_path,
            'audit/audit.log': root.with_name(root.name+'-audit.log'),
            'audit/verify_phrase_neighbours.py': auditor,
            'logs/diagnosis.log': root.with_suffix('.log'), 'logs/tests.log': tests,
            'publication/package_phrase_neighbours.py': Path(__file__)}
        for name, path in extras.items():
            assert not SECRET.search(path.read_bytes()), path
            archive.add(name, source=path)
        manifest = archive.close(); count = verify_members(output/manifest['name'])
        manifest.update(release_tag='research-hey-nyx-20261110',
            source_commit=launch['checkout_revision'], accepted=False,
            source_paths=result['source_paths'], computed_pair_distances=result['computed_pair_distances'],
            complete_selected_feature_arrays_included=True, source_pcm_and_full_cache_included=False,
            dependencies='Original source-WAV corpus and full scalar feature cache are separately byte-bound inputs; compact-broad-bank-v1.tar.gz supplies the original bank metadata.',
            listed_member_hashes_verified=count, public_download_verified=False,
            url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+manifest['name'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
