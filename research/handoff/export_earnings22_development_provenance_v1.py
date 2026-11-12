"""Export attributable text/provenance/code only, excluding all audio/features."""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    source_plan = args.sources / 'source-reservation-v1.json'
    reservation = json.loads(source_plan.read_text())
    manifest_path = args.prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_reservation_sha256'] == sha(source_plan.read_bytes())
    assert manifest['complete_recordings'] and manifest['original_media_hashes_verified']
    assert manifest['all_decoded_durations_match_native_within_one_sample']
    assert not manifest['wake_models_scored'] and not manifest['training'] and not manifest['final_data_accessed']
    assert len(manifest['records']) == len(reservation['records']) == 12
    assert [r['source_id'] for r in manifest['records']] == [r['source_id'] for r in reservation['records']]
    archive = Archive(args.output, 'earnings22-development-provenance-v1')
    for name, key in [('metadata.csv', 'upstream_metadata_sha256'), ('README.md', 'upstream_readme_sha256'),
                      ('LICENSE.md', 'upstream_license_sha256')]:
        source = args.sources / name
        assert sha(source.read_bytes()) == reservation[key]
        archive.add('sources/' + name, source=source)
    archive.add('sources/source-reservation-v1.json', source=source_plan)
    for row in reservation['records']:
        for field, expected in [('pointer_path', 'pointer_sha256'), ('transcript_path', 'transcript_sha256')]:
            source = args.sources / row[field]
            assert sha(source.read_bytes()) == row[expected]
            archive.add('sources/' + row[field], source=source)
    for name in ('manifest.json', 'runtime.json', 'protocol.json', 'preparer.py'):
        archive.add('reference/' + name, source=args.prepared / name)
    for relative, expected in [
            ('research/wakeword_p4/experiments/prepare_earnings22_development_v1.py', manifest['preparer_sha256']),
            ('research/wakeword_p4/training/features.py', json.loads((args.prepared / 'protocol.json').read_text())['frontend_source_sha256']),
            ('research/wakeword_p4/experiments/earnings22_development_preparation_protocol_v1.json', manifest['protocol_sha256'])]:
        source = repo / relative
        assert sha(source.read_bytes()) == expected
        archive.add('source/' + relative, source=source)
    archive.add('README.md', source=repo / 'research/wakeword_p4/experiments/earnings22_development_v1.md')
    archive.add('reproduce/export_earnings22_development_provenance_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    assert not any(Path(name).suffix.lower() in ('.mp3', '.wav', '.pcm', '.npy', '.npz') for name in archive.files)
    asset = archive.close()
    asset.update(
        url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
        scope='Public original text snapshots and preparation provenance/code only; no source/decoded audio or feature-array payloads.',
        source_reservation_sha256=manifest['source_reservation_sha256'],
        preparation_protocol_sha256=manifest['protocol_sha256'],
        prepared_manifest_sha256=sha(manifest_path.read_bytes()),
        source_hours=manifest['source_hours'], complete_recordings=12,
        wake_models_scored=False, training=False, final_data_accessed=False,
        attribution='Earnings22, Del Rio et al.; revdotcom/speech-datasets at ' + reservation['upstream_commit'] + '; original transcript/text license in sources/LICENSE.md.',
        excluded=['original media', 'decoded audio', 'native decoded samples', 'frontend-array payloads',
                  'models/scores', 'private recordings/credentials/device state', 'final sets'])
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
