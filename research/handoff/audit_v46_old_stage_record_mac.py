"""Recompute the original v46 historical gate from authenticated saved reports.

The missing original threshold report is an explicit input with a pinned SHA.
No raw score, feature, model or waveform arrays are loaded by this audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_v44_recorded_evidence import upper95
from audit_v46_training_record_mac import MODEL, THRESHOLDS, require
import subprocess


HERE = Path(__file__).resolve().parent
REF = '5e4d7b3e15d2a4a6879acbe4e834da33c1c1d60b'
CALIBRATION = '958ba828dac95143d924c2b620bf4d846d5f8a4cd6b0271c3f47bc62f242ff8b'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output required')
    calibration_bytes = args.historical_calibration.read_bytes()
    require(sha(calibration_bytes) == CALIBRATION, 'Original historical calibration differs')
    frozen = json.loads(calibration_bytes)
    names = {key: '4090-v46-' + tail + '.json' for key, tail in dict(
        phone='phone-development-historical-threshold-v1', vctk='vctk-development-historical-threshold-v1',
        background='voxconverse-development-v1', joint='reused-development-exploratory-calibration-v1',
        receipt='score-receipt-v1', summary='rejection-summary-v1').items()}
    raw = {key: subprocess.check_output(['git', '-C', str(HERE.parents[1]), 'show',
        REF + ':research/wakeword_p4/results/' + name]) for key, name in names.items()}
    reports = {key: json.loads(value) for key, value in raw.items()}
    links, historical = {}, {}
    for key, count in (('phone', 80), ('vctk', 300)):
        report = reports[key]
        require(report['model_sha256'] == MODEL and report['calibration_sha256'] == CALIBRATION
            and report['threshold'] == frozen['threshold'] == .99609375
            and len(report['records']) == count, 'Historical synthetic identity differs')
        rows = report['records']
        counts = dict(positive_clips=sum(item['record']['target'] == 'hey_nyx' for item in rows),
            positive_any_event_clips=sum(bool(item['nyx']['event_seconds']) for item in rows
                if item['record']['target'] == 'hey_nyx'))
        for target, group in ((None, 'common_negative'), ('hi_esp', 'cross-target_negative')):
            negatives = [item for item in rows if item['record']['target'] == target]
            counts[group] = dict(clips=len(negatives), false_events=sum(len(item['nyx']['event_seconds']) for item in negatives),
                false_activation_clips=sum(bool(item['nyx']['event_seconds']) for item in negatives))
        require(all(counts[k] == report['summaries']['nyx'][k] for k in counts if k != 'positive_clips'),
            'Saved synthetic event totals differ')
        historical[key] = counts
        for index, item in enumerate(rows):
            links[f'.local/wakeword/v46-{key}-development-historical-threshold-v1/{index}-scores.npy'] = item['nyx']['score_sha256']
    background, joint, receipt, summary = [reports[key] for key in ('background', 'joint', 'receipt', 'summary')]
    require(background['model_sha256'] == MODEL and background['frozen_report_sha256'] == CALIBRATION
        and len(background['speakers']) == 87 and background['sliding_window'] == 5, 'Background identity differs')
    for item in background['speakers']:
        require(item['false_events'] == len(item['event_seconds']), 'Saved background event count differs')
        links['.local/wakeword/v46-voxconverse-development-v1/' + item['speaker'] + '-scores.npy'] = item['score_sha256']
    require(sum(item['false_events'] for item in background['speakers']) == background['false_events'] == 3,
        'Background total differs')
    require(abs(upper95(3, background['source_hours']) - background['false_per_hour_upper_95']) < 1e-12,
        'Historical upper95 differs')
    require(len(links) == len(receipt['traces']) == receipt['authenticated_traces'] == 467
        and {item['path']: item['sha256'] for item in receipt['traces']} == links,
        'Saved report-to-receipt links differ')
    require(joint['model_sha256'] == MODEL and joint['sliding_window'] == 5
        and joint['synthetic_report_sha256'] == [sha(raw[key]) for key in ('phone', 'vctk')]
        and joint['background_report_sha256'] == sha(raw['background']), 'Joint report inputs differ')
    grid = joint['grid']
    require([r['threshold'] for r in grid] == sorted(set(THRESHOLDS + [.5 + i / 256 for i in range(128)])),
        'Original 138-point grid differs')
    max_error = 0.
    for point in grid:
        groups = list(point['groups'].values())
        macro = sum(group['triggered'] / group['positives'] for group in groups) / len(groups)
        require(abs(macro - point['synthetic_macro_clip_trigger']) < 1e-15, 'Joint macro differs')
        upper = upper95(point['background_events'], point['background_hours'])
        max_error = max(max_error, abs(upper - point['background_upper_95']))
        eligible = upper <= .5 and macro >= .48125 and all(
            group['false_activation_clips'] == 0 and group['pre_generation_positive_events'] == 0 for group in groups)
        require(eligible == point['eligible'], 'Joint eligibility differs')
    require(max_error < 1e-12 and len(grid) == summary['grid_points'] == 138
        and sum(r['eligible'] for r in grid) == summary['eligible'] == 0, 'Joint rejection differs')
    minimum_false = min(sum(g['false_activation_clips'] for g in r['groups'].values()) for r in grid)
    maximum_macro = max(r['synthetic_macro_clip_trigger'] for r in grid)
    require(minimum_false == summary['min_negative_false_clips'] == 4
        and maximum_macro == summary['max_macro'] == .7458333333333333, 'Joint extrema differ')
    require(joint['threshold'] is None and not joint['viable_operating_point']
        and not summary['new_development_accessed'] and not summary['final_data_used'], 'Rejection record differs')
    result = dict(scope=__doc__, gpu_commit=REF, model_sha256=MODEL,
        input_report_sha256={names[key]: sha(value) for key, value in raw.items()},
        original_historical_calibration_sha256=CALIBRATION, score_receipt_report_links_checked=len(links),
        historical_saved_event_counts=historical, historical_background_events=3,
        historical_background_hours=background['source_hours'], historical_background_upper95=background['false_per_hour_upper_95'],
        joint_grid_points_recomputed=138, joint_eligible_points=0, minimum_negative_false_clips=minimum_false,
        maximum_macro=maximum_macro, maximum_poisson_upper95_absolute_difference=max_error,
        script_sha256=sha(Path(__file__).read_bytes()), all_checks_pass=True, accepted=False,
        poisson_implementation_sha256=sha((HERE / 'audit_v44_recorded_evidence.py').read_bytes()),
        limits='Recorded report/hash-link/count arithmetic only. Mac does not independently rehash/replay the GPU score files, weights, model or feature arrays. No unused-development/final, audio, P4 operation or parity proof. Original rejection remains in force.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
