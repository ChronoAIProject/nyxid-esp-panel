"""Package a completed, frozen historical v30 software-development baseline."""
import argparse
import json
from pathlib import Path

from export_assets import Archive, encoded, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'model-dir', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    report = json.loads((args.study / 'report.json').read_text())
    protocol = json.loads((args.study / 'protocol.json').read_text())
    assert sha((args.study / 'protocol.json').read_bytes()) == report['protocol_sha256']
    assert sha((args.study / 'scores.npz').read_bytes()) == report['scores_sha256']
    assert report['all_reset_replays_exact'] and report['all_sources_hash_verified']
    assert len(report['records']) == 126 and not report['threshold_selected']
    assert report['accepted'] is None and report['far_estimate'] is None
    archive = Archive(args.output, 'bryce-v30-historical-baseline-v1')
    for name in ('protocol.json', 'source.py', 'scores.npz', 'report.json'):
        archive.add('study/' + name, source=args.study / name)
    for name, field in [('hey_nyx.tflite', 'model_sha256'), ('training.json', 'training_report_sha256'),
                        ('4090-v30-expanded-calibration.json', 'calibration_sha256')]:
        source = args.model_dir / name
        assert sha(source.read_bytes()) == protocol['v30'][field]
        archive.add('model/' + name, source=source)
    scorer = 'research/wakeword_p4/experiments/score_bryce_v30_baseline_v1.py'
    paths = {**protocol['helper_source_sha256'], scorer: protocol['v30_scorer_sha256']}
    for relative, expected in paths.items():
        source = repo / relative
        assert sha(source.read_bytes()) == expected
        archive.add('source/' + relative, source=source)
    archive.add('source/research/wakeword_p4/experiments/bryce_historical_baseline_protocol_v1.json',
                source=args.study / 'protocol.json')
    archive.add('README.md', source=repo / 'research/wakeword_p4/experiments/bryce_v30_historical_baseline_v1.md')
    archive.add('reproduce/export_bryce_v30_baseline_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    asset = archive.close()
    asset.update(
        url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
        scope='Historical v30 on126new development WAVs; no Hi ESP result, FAR, current GPU candidate or P4 acceptance.',
        protocol_sha256=report['protocol_sha256'], report_sha256=sha((args.study / 'report.json').read_bytes()),
        scores_sha256=report['scores_sha256'], model_sha256=report['model_sha256'])
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
