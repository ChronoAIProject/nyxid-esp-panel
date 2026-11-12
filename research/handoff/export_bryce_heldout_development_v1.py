"""Export the frozen saved Kristin/Norman development cohort, never regenerate it."""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'models', 'audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    experiment = repo / 'research/wakeword_p4/experiments'
    data_manifest = json.loads((args.data / 'manifest.json').read_text())
    plan = json.loads((args.data / 'plan.json').read_text())
    audit = json.loads(args.audit.read_text())
    assert audit['manifest_sha256'] == sha((args.data / 'manifest.json').read_bytes())
    assert audit['plan_sha256'] == sha((args.data / 'plan.json').read_bytes())
    assert audit['native_files_verified'] == 42 and audit['wav_files_verified'] == 126
    assert audit['independently_reconstructed_wav_bytes_exact']
    assert not audit['wake_models_scored'] and not audit['training_data']
    generator = experiment / 'generate_bryce_heldout_development_v1.py'
    auditor = experiment / 'audit_bryce_heldout_development_v1.py'
    assert audit['auditor_sha256'] == sha(auditor.read_bytes())
    assert audit['generator_sha256'] == sha(generator.read_bytes())
    archive = Archive(args.output, 'heldout-bryce-development-v1')
    for name in ('plan.json', 'manifest.json'):
        archive.add('data/' + name, source=args.data / name)
    for row in data_manifest['native_records']:
        source = args.data / row['native_path']
        assert source.name == row['native_path'] and sha(source.read_bytes()) == row['native_sha256']
        archive.add('data/' + source.name, source=source)
    for row in data_manifest['records']:
        source = args.data / row['path']
        assert source.name == row['path'] and sha(source.read_bytes()) == row['sha256']
        archive.add('data/' + source.name, source=source)
    for voice in plan['voices']:
        name = voice['name'] + '.onnx.json'
        source = args.models / name
        assert sha(source.read_bytes()) == voice['files'][name]
        archive.add('models/' + name, source=source)
    for source in (generator, auditor):
        archive.add('reproduce/' + source.name, source=source)
    archive.add('independent-audit-v1.json', source=args.audit)
    archive.add('README.md', source=experiment / 'bryce_heldout_development_v1.md')
    lock = ''.join(f'{key}=={value}\n' for key, value in sorted(plan['versions'].items()))
    archive.add('reproduce/requirements-lock.txt', data=lock.encode())
    archive.add('reproduce/export_bryce_heldout_development_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    asset = archive.close()
    asset.update(
        url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
        scope='Synthetic source-heldout development only; no training, final testing, wake-model scores or device data.',
        generation_plan_sha256=audit['plan_sha256'], data_manifest_sha256=audit['manifest_sha256'],
        native_sources=42, rendered_wavs=126, wake_models_scored=False,
        excluded=['redownloadable ONNX models', 'venv/dependency caches', 'private microphone/room data',
                  'device state/credentials/firmware', 'existing final sets'])
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
