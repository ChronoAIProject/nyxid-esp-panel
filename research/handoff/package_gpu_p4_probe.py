"""Package the completed silent v30 P4 probe, excluding device state and raw UART."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive, SECRET
from fetch_assets import verify_archive

NAME = 'gpu-v30-p4-probe-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        name = 'CONTENTS-'+NAME+'.json'
        inventory = json.loads(bundle.extractfile(name).read())
        assert set(bundle.getnames()) == set(inventory)|{name}
        for name, expected in inventory.items():
            data = bundle.extractfile(name).read()
            assert len(data) == expected['bytes']
            assert hashlib.sha256(data).hexdigest() == expected['sha256']
        return len(inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-public', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if args.verify_public:
        manifest = json.loads((output/'manifest.json').read_bytes())
        download = output/'anonymous-download.tar.gz'
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'nyx-p4-research'})
        with urllib.request.urlopen(request, timeout=60) as response, download.open('xb') as out:
            while block := response.read(1024*1024):
                out.write(block)
        assert download.stat().st_size == manifest['bytes'] and sha(download) == manifest['sha256']
        assert verify(download) == manifest['listed_member_hashes_verified']
        manifest.update(public_download_verified=True, anonymous_download_sha256=sha(download))
    else:
        if args.artifacts is None:
            parser.error('--artifacts is required to build')
        root = args.artifacts.resolve(); repo = Path(__file__).resolve().parents[2]
        for directory, headroom in (('board', 'headroom-private.json'),
                                    ('board-hiesp', 'hiesp-headroom-private.json')):
            runner = json.loads((root/directory/'report.json').read_bytes())
            idle = json.loads((root/headroom).read_bytes())
            assert runner['restored'] and runner['acceptance_exit_code'] == 0
            assert idle['completed'] and idle['capture']['passed']
        result = json.loads((root/'result.json').read_bytes())
        assert result['accepted'] is False
        receipt = json.loads((root/'receipt-audit.json').read_bytes())
        assert receipt['all_four_members_verified'] and receipt['all_three_payload_hashes_match_frozen_mac_profile']
        verdict = json.loads((root/'selftest-verdict-final.json').read_bytes())
        assert verdict['uart_sha256'] == sha(root/'board/uart-private.log')
        assert verdict['outputs'] == 179 and verdict['mismatches'] == 0 and verdict['activation_steps'] == [144]
        output.mkdir(parents=True, exist_ok=False); archive = Archive(output, NAME)
        files = {
            'reports/result.json': 'result.json',
            'reports/receipt.json': 'receipt-audit.json',
            'reports/selftest.json': 'selftest-verdict-final.json',
            'reports/selftest-startup-prefix.json': 'selftest-verdict.json',
            'reports/selftest-sanitized.json': 'selftest-verdict-sanitized.json',
            'reports/v30-idle.json': 'headroom-private.json',
            'reports/hiesp-idle.json': 'hiesp-headroom-private.json',
            'reports/v30-runner.json': 'board/report.json',
            'reports/hiesp-runner.json': 'board-hiesp/report.json',
            'counts/v30.txt': 'v30-counts-only.txt',
            'counts/hiesp.txt': 'hiesp-counts-only.txt',
            'counts/selftest-markers.txt': 'selftest-markers-only.txt',
            'fixture/selftest.bin': 'v30-prime-selftest.bin',
            'fixture/selftest.json': 'v30-prime-selftest.json',
            'diagnostic/plan.json': 'diagnostic-training-sources-plan.json',
            'diagnostic/result.json': 'diagnostic-training-source-results.json',
            'diagnostic/scores.npz': 'diagnostic-training-source-scores.npz',
            'diagnostic/replay-source.py': 'reproduced-training-source-diagnostic/source.py',
            'diagnostic/legacy-source-rejection.json': 'legacy-source-rejection-diagnostic.json',
            'diagnostic/legacy-source-arrays.npz': 'legacy-source-rejection-arrays.npz',
            'build/model-binding.json': 'v30-build-manifest.json',
            'build/calculator-warning-exception.json': 'calculator-build-exception.json',
            'model/hey_nyx.tflite': 'payload/hey_nyx.tflite',
            'model/training.json': 'payload/training.json',
            'model/calibration.json': 'payload/4090-v30-expanded-calibration.json',
            'model/original-handoff-manifest.json': 'payload/manifest.json',
        }
        for name, relative in files.items():
            path = root/relative
            assert path.is_file() and not path.is_symlink()
            archive.add(name, source=path)
        for relative in ('audit_runtime_selftest.py', 'test_audit_runtime_selftest.py',
                         'diagnose_gpu_sources.py', 'summarize_gpu_p4_probe.py',
                         'run_full_app.py', 'measure_idle_headroom.py',
                         'prepare_runtime.py', 'plans/gpu-v30-runtime-v1.json',
                         'dependency_pins/idf_component.yml'):
            path = repo/'research/panel_wake_runtime'/relative
            assert not SECRET.search(path.read_bytes())
            archive.add('source/research/panel_wake_runtime/'+relative, source=path)
        for relative in ('prepare_runtime_selftest.py', 'evaluate.py', 'features.py', 'detector_policy.py'):
            archive.add('source/research/wakeword_p4/training/'+relative,
                        source=repo/'research/wakeword_p4/training'/relative)
        for relative in ('package_gpu_p4_probe.py', 'export_assets.py', 'fetch_assets.py'):
            archive.add('source/research/handoff/'+relative, source=Path(__file__).parent/relative)
        archive.add('source/dependencies.lock', source=repo/'dependencies.lock')
        manifest = archive.close()
        manifest.update(listed_member_hashes_verified=verify(output/manifest['name']),
            release_tag='research-hey-nyx-20261110', accepted=False,
            public_download_verified=False,
            scope='Silent numeric self-test and matched steady idle resource probe; no acoustic acceptance.',
            synthetic_pcm_fixture_included=True, device_state_firmware_raw_uart_and_microphone_audio_included=False,
            url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+manifest['name'])
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
