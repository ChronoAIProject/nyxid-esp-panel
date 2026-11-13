"""Verify both published recovery captures and replay the unchanged comparator."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


ARCHIVES = {
    'gpu-complete-recovery-integration-v1':
        (8397595, '5116044fd41a89caf63c83cce5cf24457c315482e5dafdbbcd53ce2dba0357e1'),
    'gpu-complete-recovery-cuda-integration-v1':
        (7827074, '75653fbd4266f5cbd0f543cb61ad2471a80a0dd9474a42759663ad0fb79ac0b4')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--received', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    comparator = Path(__file__).with_name('compare_complete_recovery_integration_v1.py')
    receipts = []
    for name, (size, digest) in ARCHIVES.items():
        archive, root = args.received / (name + '.tar.gz'), args.received / name
        assert archive.stat().st_size == size and sha(archive) == digest
        manifest = json.loads((root / 'manifest.json').read_text())
        entries = manifest['files']
        names = [e['path'] for e in entries]
        assert len(names) == len(set(names)) and 'manifest.json' not in names
        with tarfile.open(archive) as source:
            members = source.getmembers()
            assert all(m.isfile() for m in members)
            assert len(members) == len(entries) + 1
            assert {m.name for m in members} == set(names) | {'manifest.json'}
            for entry in entries:
                relative = Path(entry['path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                p = root / relative
                assert p.stat().st_size == entry['size'] and sha(p) == entry['sha256']
                assert source.extractfile(entry['path']).read() == p.read_bytes()
        derived = args.received / (name + '-mac-comparison.json')
        result = subprocess.run([sys.executable, str(comparator), '--original', str(root / 'original'),
                                 '--disabled', str(root / 'disabled'), '--enabled', str(root / 'enabled'),
                                 '--output', str(derived)], check=True, capture_output=True, text=True)
        assert result.returncode == 0
        assert derived.read_bytes() == (root / 'comparison.json').read_bytes()
        receipts.append(dict(name=name, archive_bytes=size, archive_sha256=digest,
                             payload_count=len(entries), all_payload_hashes_exact=True,
                             unchanged_mac_comparator_passed=True,
                             comparison_byte_exact=True, comparison_sha256=sha(derived),
                             comparison=json.loads(derived.read_text())))
    report = dict(schema='mac-gpu-complete-recovery-independent-receipt-v1',
                  source_sha256=sha(Path(__file__)), comparator_sha256=sha(comparator),
                  captures=receipts, training_reexecuted=False,
                  acceptance_claim=False,
                  limits='Anonymous published capture integrity and independent comparator replay only. '
                         'Mac did not rerun full training or authenticate all original missing feature arrays. '
                         'Two updates do not establish model quality; no new detector promoted.')
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(payloads_verified=[r['payload_count'] for r in receipts],
                          all_comparisons_byte_exact=True)))


if __name__ == '__main__':
    main()
