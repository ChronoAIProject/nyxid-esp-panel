"""Publish saved phoneme observations without audio, model weights or device state."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive, SECRET
from fetch_assets import verify_archive

NAME = 'review-phoneme-consistency-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_members(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        name = 'CONTENTS-'+NAME+'.json'
        inventory = json.loads(bundle.extractfile(name).read())
        assert set(bundle.getnames()) == set(inventory)|{name}
        for member, expected in inventory.items():
            data = bundle.extractfile(member).read()
            assert len(data) == expected['bytes'], member
            assert hashlib.sha256(data).hexdigest() == expected['sha256'], member
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
        root = args.artifacts.resolve()
        result = json.loads((root/'result.json').read_bytes())
        plan = json.loads((root/'plan.json').read_bytes())
        audit_path = root.with_name(root.name+'-audit.json')
        audit = json.loads(audit_path.read_bytes())
        auditor = Path(__file__).with_name('verify_review_phonemes.py')
        assert audit['result_sha256'] == sha(root/'result.json') and result['accepted'] is False
        assert audit['auditor_sha256'] == sha(auditor)
        assert audit['sources'] == result['source_paths'] == plan['source_paths']
        assert audit['triplets'] == result['review_triplets'] == plan['review_triplets']
        for field in ('all_original_wav_hashes_rates_and_lengths_verified',
                      'all_logits_and_greedy_ids_phone_strings_verified',
                      'all_edit_orders_cohorts_and_prediction_collisions_reconstructed'):
            assert audit[field]
        assert result['plan_sha256'] == sha(root/'plan.json')
        for name, expected in result['evidence_sha256'].items():
            assert sha(root/name) == expected
        for name, expected in plan['source_sha256'].items():
            assert sha(root/'source'/name) == expected
        for name, expected in plan['model_files_sha256'].items():
            if name.endswith('.json'):
                assert sha(root/'model-config'/name) == expected
        tests = root.with_name('review-phoneme-consistency-preflight.log')
        assert 'Ran 3 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
        output.mkdir(parents=True, exist_ok=False); archive = Archive(output, NAME)
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            assert not path.is_symlink() and path.suffix in ('.py', '.json', '.jsonl', '.npz')
            archive.add('study/'+path.relative_to(root).as_posix(), source=path)
        extras = {'audit/audit.json': audit_path,
            'audit/audit.log': root.with_name(root.name+'-audit.log'),
            'audit/verify_review_phonemes.py': auditor,
            'logs/inference.log': root.with_suffix('.log'), 'logs/tests.log': tests,
            'publication/package_review_phonemes.py': Path(__file__)}
        for name, path in extras.items():
            assert not SECRET.search(path.read_bytes()), path
            archive.add(name, source=path)
        manifest = archive.close(); count = verify_members(output/manifest['name'])
        manifest.update(release_tag='research-hey-nyx-20261110',
            source_commit=result['source_commit'], accepted=False,
            source_paths=result['source_paths'], review_triplets=result['review_triplets'],
            complete_frame_logits_included=True,
            source_pcm_model_weights_and_private_listening_included=False,
            dependencies='Pinned public phoneme weights, original source-WAV corpus and feature index; phrase-neighbours-v1.tar.gz supplies the frozen inventory/review queue. All identities are bound in study/plan.json.',
            listed_member_hashes_verified=count, public_download_verified=False,
            url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+manifest['name'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
