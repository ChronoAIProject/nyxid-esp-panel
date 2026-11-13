"""Package the completed continuation diagnostic with original saved traces."""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'baseline', 'model', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    report = json.loads((args.study / 'report.json').read_text())
    protocol = json.loads((args.study / 'protocol.json').read_text())
    assert not report['acceptance_claim'] and report['source_clips'] == 126
    assert sha((args.study / 'scores.npz').read_bytes()) == report['scores_sha256']
    assert sha((args.study / 'source.py').read_bytes()) == protocol['source_sha256'] == report['source_sha256']
    archive = Archive(args.output, 'bryce-same-prefix-continuation-v1')
    for name in ('report.json', 'protocol.json', 'source.py', 'scores.npz', 'audit.json'):
        archive.add('study/' + name, source=args.study / name)
    for name, key in (('report.json', 'baseline_report_sha256'), ('scores.npz', 'baseline_scores_sha256')):
        assert sha((args.baseline / name).read_bytes()) == protocol[key]
        archive.add('baseline/' + name, source=args.baseline / name)
    assert sha(args.model.read_bytes()) == protocol['model_sha256']
    archive.add('model/hey_nyx.tflite', source=args.model)
    paths = dict(protocol['helper_source_sha256'])
    paths['research/wakeword_p4/experiments/bryce_same_prefix_continuation_v1.py'] = protocol['source_sha256']
    for relative, expected in paths.items():
        source = repo / relative
        assert sha(source.read_bytes()) == expected
        archive.add('source/' + relative, source=source)
    for name in ('audit_bryce_same_prefix_scores_v1.py', 'diagnose_bryce_v30_saved_scores_v1.py'):
        archive.add('source/research/wakeword_p4/experiments/' + name,
                    source=repo / 'research/wakeword_p4/experiments' / name)
    archive.add('README.md', source=repo / 'research/wakeword_p4/experiments/bryce_same_prefix_continuation_v1.md')
    archive.add('source-data-asset.json', source=repo / 'research/handoff/assets-heldout-bryce-development-v1.json')
    archive.add('reproduce/export_bryce_same_prefix_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    asset = archive.close()
    asset.update(url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
                 scope='Consumed development: identical-prefix dry-tail intervention on frozen v30; no promotion.',
                 report_sha256=sha((args.study / 'report.json').read_bytes()),
                 protocol_sha256=report['protocol_sha256'], scores_sha256=report['scores_sha256'],
                 model_sha256=report['model_sha256'])
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
