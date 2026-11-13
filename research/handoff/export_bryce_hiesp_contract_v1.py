"""Export replay source/metadata/audits, without duplicating existing WAV assets."""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'audit', 'base-runner', 'patched-runner', 'rejected-audit', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    audit = json.loads(args.audit.read_text())
    integrity = json.loads((args.inputs / 'integrity.json').read_text())
    assert audit['passed'] and not audit['qemu_invoked'] and not audit['actual_gpu_engine_authenticated']
    assert audit['manifest_sha256'] == integrity['manifest_sha256'] == sha((args.inputs / 'manifest.json').read_bytes())
    assert audit['base_runner_sha256'] == sha(args.base_runner.read_bytes())
    assert audit['patched_runner_sha256'] == sha(args.patched_runner.read_bytes())
    archive = Archive(args.output, 'bryce-hiesp-replay-contract-v1')
    for name in ('manifest.json', 'integrity.json'):
        archive.add('reference/' + name, source=args.inputs / name)
    for name in ('source-manifest.json', 'generation-plan.json', 'source-audit.json'):
        archive.add('reference/provenance/' + name, source=args.inputs / 'provenance' / name)
    archive.add('reference/base-run_replay.py', source=args.base_runner)
    archive.add('reference/contract-audit.json', source=args.audit)
    archive.add('history/rejected-draft-audit.json', source=args.rejected_audit)
    archive.add('history/rejected-draft-runner.py', source=args.rejected_audit.parent / 'patched-runner.py')
    replay_prefix = 'source/research/wakeword_p4/qemu_esp32_replay/'
    archive.add(replay_prefix + 'run_replay.py', source=args.patched_runner)
    archive.add(replay_prefix + 'synthetic_development.py', source=repo / 'research/wakeword_p4/qemu_esp32_replay/synthetic_development.py')
    prefix = 'source/research/wakeword_p4/experiments/'
    for name in ('prepare_bryce_hiesp_replay_v1.py', 'freeze_bryce_hiesp_replay_v1.py',
                 'audit_bryce_hiesp_replay_contract_v1.py', 'bryce_hiesp_development_runner_v1.patch'):
        archive.add(prefix + name, source=repo / 'research/wakeword_p4/experiments' / name)
    archive.add('README.md', source=repo / 'research/wakeword_p4/experiments/bryce_hiesp_replay_contract_v1.md')
    archive.add('reproduce/export_bryce_hiesp_contract_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    assert not any(Path(name).suffix.lower() in ('.wav', '.mp3', '.pcm', '.npy', '.npz', '.bin', '.elf') for name in archive.files)
    asset = archive.close()
    asset.update(
        url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
        scope='Frozen source/metadata/admission tests only. No actual QEMU/WakeNet inference, engine binaries, model scores or hardware acceptance.',
        source_wav_asset='heldout-bryce-development-v1.tar.gz',
        derived_manifest_sha256=audit['manifest_sha256'], base_runner_sha256=audit['base_runner_sha256'],
        patched_runner_sha256=audit['patched_runner_sha256'], validator_sha256=audit['validator_sha256'],
        contract_audit_sha256=sha(args.audit.read_bytes()), all_contract_cases_passed=True,
        actual_gpu_engine_authenticated=False, qemu_invoked=False, wake_models_scored=False)
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
