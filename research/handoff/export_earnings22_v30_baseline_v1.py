"""Export historical development score outputs/code, never source audio/features."""
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
    assert report['recordings'] == len(report['records']) == 12
    assert report['all_reset_replays_exact'] and report['all_sources_authenticated_before_model_load']
    assert report['accepted'] is None and not report['threshold_selected']
    assert not report['current_gpu_candidate_scored'] and not report['final_data_accessed']
    archive = Archive(args.output, 'earnings22-v30-historical-baseline-v1')
    for name in ('protocol.json', 'source.py', 'scores.npz', 'report.json'):
        archive.add('study/' + name, source=args.study / name)
    for name, field in [('hey_nyx.tflite', 'model_sha256'), ('training.json', 'training_report_sha256'),
                        ('4090-v30-expanded-calibration.json', 'calibration_sha256')]:
        source = args.model_dir / name
        assert sha(source.read_bytes()) == protocol['v30'][field]
        archive.add('model/' + name, source=source)
    scorer = 'research/wakeword_p4/experiments/score_earnings22_v30_baseline_v1.py'
    for relative, expected in {**protocol['helper_source_sha256'], scorer: protocol['scorer_sha256']}.items():
        source = repo / relative
        assert sha(source.read_bytes()) == expected
        archive.add('source/' + relative, source=source)
    archive.add('source/research/wakeword_p4/experiments/earnings22_historical_v30_score_protocol_v1.json',
                source=args.study / 'protocol.json')
    archive.add('README.md', source=repo / 'research/wakeword_p4/experiments/earnings22_v30_historical_baseline_v1.md')
    archive.add('reproduce/export_earnings22_v30_baseline_v1.py', source=Path(__file__))
    archive.add('reproduce/export_assets.py', source=repo / 'research/handoff/export_assets.py')
    asset = archive.close()
    asset.update(
        url='https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/' + asset['name'],
        scope='Historical v30 software-reference scores on12complete development calls; no source audio/frontend payload, Hi ESP, current GPU candidate or final acceptance.',
        protocol_sha256=report['protocol_sha256'], report_sha256=sha((args.study / 'report.json').read_bytes()),
        scores_sha256=report['scores_sha256'], model_sha256=report['model_sha256'], source_hours=report['source_hours'],
        negative_intended_background_events=report['negative_intended_background_events'])
    (args.output / 'asset.json').write_bytes(encoded(asset))
    print(json.dumps(asset))


if __name__ == '__main__':
    main()
