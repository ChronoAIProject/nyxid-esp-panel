"""Publish the frozen v30 Mac regression evidence without private device data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from export_assets import Archive
from fetch_assets import verify_archive

NAME = 'gpu-v30-mac-regression-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(path):
    verify_archive(path)
    with tarfile.open(path) as bundle:
        name = 'CONTENTS-'+NAME+'.json'
        entries = json.loads(bundle.extractfile(name).read())
        assert set(bundle.getnames()) == set(entries)|{name}
        for name, expected in entries.items():
            raw = bundle.extractfile(name).read()
            assert len(raw) == expected['bytes']
            assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    return len(entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'sources', 'model-dir'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-public', action='store_true')
    args = parser.parse_args(); out = args.output
    if args.verify_public:
        manifest = json.loads((out/'manifest.json').read_bytes())
        download = out/'anonymous-download.tar.gz'
        request = urllib.request.Request(manifest['url'], headers={'User-Agent': 'nyx-wake-research'})
        with urllib.request.urlopen(request, timeout=60) as response, download.open('xb') as dest:
            while block := response.read(1024*1024):
                dest.write(block)
        assert sha(download) == manifest['sha256'] and download.stat().st_size == manifest['bytes']
        assert verify(download) == manifest['listed_member_hashes_verified']
        manifest.update(public_download_verified=True, anonymous_download_sha256=sha(download))
    else:
        if not all((args.artifacts, args.sources, args.model_dir)):
            parser.error('Build requires --artifacts, --sources and --model-dir')
        root = args.artifacts; repo = Path(__file__).resolve().parents[2]
        result = json.loads((root/'result.json').read_bytes())
        audit = json.loads((root/'audit.json').read_bytes())
        assert result['accepted'] is False and audit['passed'] and result['sources'] == 350
        assert audit['result_sha256'] == sha(root/'result.json')
        assert audit['scores_sha256'] == sha(root/'scores.npz')
        out.mkdir(parents=True, exist_ok=False); archive = Archive(out, NAME)
        for name in ('result.json', 'scores.npz', 'plan.json', 'audit.json', 'source.py', 'auditor-source.py'):
            archive.add('study/'+name, source=root/name)
        for name in ('baseline.json', 'manifest.jsonl', 'provenance.json'):
            archive.add('inputs/'+name, source=args.sources/name)
        for name in ('hey_nyx.tflite', 'training.json', '4090-v30-expanded-calibration.json'):
            archive.add('model/'+name, source=args.model_dir/name)
        for relative in (
                'wakeword_p4/training/replay_gpu_mac_regression.py',
                'wakeword_p4/training/verify_gpu_mac_regression.py',
                'wakeword_p4/training/evaluate.py', 'wakeword_p4/training/features.py',
                'wakeword_p4/training/detector_policy.py',
                'wakeword_p4/training/prepare_runtime_selftest.py',
                'wakeword_p4/training/plans/gpu-v30-mac-regression-v1.json',
                'panel_wake_runtime/plans/gpu-v30-runtime-v1.json',
                'wakeword_baseline/compare.py', 'handoff/package_gpu_mac_regression.py',
                'handoff/export_assets.py', 'handoff/fetch_assets.py'):
            archive.add('source/research/'+relative, source=repo/'research'/relative)
        manifest = archive.close()
        manifest.update(listed_member_hashes_verified=verify(out/manifest['name']),
            release_tag='research-hey-nyx-20261110', accepted=False,
            scope=result['scope'], public_download_verified=False,
            external_source_asset='paired-mac-regression-v1.tar.gz',
            source_pcm_firmware_microphone_audio_and_device_state_included=False,
            url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/'+manifest['name'])
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
