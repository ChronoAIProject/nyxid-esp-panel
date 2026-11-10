"""Select a threshold using only existing validation traces from several corpora.

Keeps each negative corpus separate. Zero observed events is a selection gate,
not a confidence guarantee or independent-final acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from evaluate import event_indices


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count(scores, threshold):
    return len(event_indices(scores, threshold)) if np.max(scores) >= threshold else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--english-validation', type=Path, required=True)
    p.add_argument('--negative-validation', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report path')
    base = json.loads((a.english_validation/'report.json').read_text())
    calibration = json.loads((a.english_validation/'threshold.json').read_text())
    if not base.get('validation_only') or base['testing_regression'] is not None:
        raise ValueError('Base calibration must contain validation only')
    if calibration['selected'] != base['validation']:
        raise ValueError('Base grid does not match report')
    window = base.get('sliding_window', 5)
    corpora, sources, seen_speakers = [], [], set()
    for directory in a.negative_validation:
        report_path = directory/'report.json'
        report = json.loads(report_path.read_text())
        if (report['model_sha256'] != base['model_sha256']
                or report.get('split', 'validation') != 'validation'
                or report.get('sliding_window', 5) != window
                or report.get('development_regression_scored', False)
                or report.get('independent_final_scored', False)
                or report.get('testing_or_independent_final_scored', False)
                or 'validation' not in report['role']):
            raise ValueError('Extra score source must be matching-model development validation')
        arrays = []
        for row in report['speakers']:
            if row['speaker'] in seen_speakers:
                raise ValueError('Validation speaker repeated across corpora')
            seen_speakers.add(row['speaker'])
            path = directory/(row['speaker']+'-scores.npy')
            if sha(path) != row['score_sha256']:
                raise ValueError('Score trace changed')
            values = np.load(path, allow_pickle=False)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError('Invalid score trace')
            if count(values, report['threshold']) != row['false_events']:
                raise ValueError('Frozen event count not reproduced')
            arrays.append(values)
        hours = sum(row['source_seconds'] for row in report['speakers'])/3600
        if not np.isclose(hours, report['source_hours'], rtol=0, atol=1e-12):
            raise ValueError('Exposure differs from frozen report')
        corpora.append((directory.name, hours, arrays))
        sources.append(dict(name=directory.name, report_sha256=sha(report_path),
                            feature_index_sha256=report['feature_index_sha256']))
    grid = []
    for original in calibration['grid']:
        threshold = original['threshold']
        negative = []
        for name, hours, arrays in corpora:
            events = sum(count(values, threshold) for values in arrays)
            negative.append(dict(corpus=name, source_hours=hours, false_events=events,
                                 false_per_hour=events/hours,
                                 false_per_hour_upper_95=float(.5*chi2.ppf(.95,2*(events+1))/hours)))
        eligible = (original['ambient_false_events'] == 0 and original['max_source_false_rate'] <= .01
                    and all(row['false_events'] == 0 for row in negative))
        grid.append(dict(english=original, negative_corpora=negative, eligible=eligible))
    eligible = [r for r in grid if r['eligible']]
    selected = max(eligible, key=lambda r: (r['english']['macro_recall'],
                                            -r['english']['max_source_false_rate'],
                                            -r['english']['threshold'])) if eligible else None
    result = dict(role='Expanded development-validation calibration; no regression/final selection',
                  model_sha256=base['model_sha256'], sliding_window=window, previous_threshold=base['threshold'],
                  threshold=selected['english']['threshold'] if selected else None,
                  selected=selected, viable_operating_point=bool(eligible), grid=grid,
                  source_sha256=sha(Path(__file__)), detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  base_report_sha256=sha(a.english_validation/'report.json'),
                  base_threshold_grid_sha256=sha(a.english_validation/'threshold.json'),
                  negative_validation_sources=sources, validation_only=True,
                  development_regression_used_for_selection=False, independent_final_scored=False,
                  selection='Require zero ambient events separately in original English and every added validation corpus, max confusing-source rate <=1%, then maximize English positive source-macro event recall.',
                  limits='Repeated development validation and threshold selection invalidate an independent acceptance interpretation. Nominal Poisson bounds describe each corpus only. Short far-field exposure and synthetic positives remain insufficient; no Hi ESP comparison.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'grid'}, indent=2))


if __name__ == '__main__':
    main()
