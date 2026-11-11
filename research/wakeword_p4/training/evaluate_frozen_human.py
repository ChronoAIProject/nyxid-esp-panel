"""Score new human validation streams at a previously frozen model/threshold."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import chi2
from evaluate import Detector, event_indices, smooth


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--frozen-report', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus-name', default='Mandarin',
                   help='Descriptive corpus label only; does not alter evaluation')
    p.add_argument("--split", choices=("validation", "testing"), default="validation",
                   help="testing is the pre-existing development regression, never independent final")
    p.add_argument('--identity-unit', choices=('speaker','recording'), default='speaker',
                   help='Interpret the legacy speaker-key identifiers explicitly')
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh evaluation directory')
    frozen = json.loads(a.frozen_report.read_text())
    if sha(a.model) != frozen['model_sha256']:
        raise ValueError('Model differs from frozen calibration')
    threshold = frozen['threshold']
    window = frozen.get('sliding_window', 5)
    cache = json.loads((a.features/'cache.json').read_text())
    if sha(a.features/'human.json') != cache['index_sha256']:
        raise ValueError('Changed stream index')
    rows = [r for r in json.loads((a.features/'human.json').read_text()) if r['split'] == a.split]
    if not rows or len({r['speaker'] for r in rows}) != len(rows):
        raise ValueError('Need unique development stream identifiers')
    a.output.mkdir(parents=True)
    detector, results, started = Detector(a.model), [], time.monotonic()
    for row in rows:
        path = (a.features/row['path']).resolve()
        if not path.is_relative_to(a.features.resolve()) or sha(path) != row['feature_sha256']:
            raise ValueError('Changed or escaped stream features')
        data = np.load(path, mmap_mode='r')
        if data.dtype != np.uint16 or data.shape != (row['frames'], 40):
            raise ValueError('Unexpected feature format')
        detector.reset(); prefix = detector.predict(data[:300])
        detector.reset(); np.testing.assert_array_equal(prefix, detector.predict(data[:300]))
        detector.reset(); values = smooth(detector.predict(data), window)
        events = event_indices(values, threshold)
        target = a.output/(row['speaker']+'-scores.npy'); np.save(target, values)
        result = dict(speaker=row['speaker'], source_seconds=row['source_seconds'],
                      features_sha256=row['feature_sha256'], score_sha256=sha(target),
                      false_events=len(events), event_seconds=[float((3*(i+window-1)+5)*.01) for i in events],
                      three_second_prefix_reset_reproducible=True)
        results.append(result)
        print(json.dumps(dict(speaker=row['speaker'], false_events=len(events),
                              completed=len(results), planned=len(rows))), flush=True)
    hours = sum(r['source_seconds'] for r in results)/3600
    count = sum(r['false_events'] for r in results)
    report = dict(role=f'{a.corpus_name} development {a.split} at frozen prior English calibration', split=a.split,
                  model_sha256=frozen['model_sha256'], threshold=threshold, sliding_window=window, threshold_reselected=False,
                  frozen_report_sha256=sha(a.frozen_report), feature_cache_sha256=sha(a.features/'cache.json'),
                  feature_index_sha256=cache['index_sha256'], source_sha256=sha(Path(__file__)),
                  detector_source_sha256=sha(Path(__file__).with_name('evaluate.py')),
                  source_hours=hours, false_events=count, false_per_hour=count/hours,
                  false_per_hour_upper_95=float(.5*chi2.ppf(.95, 2*(count+1))/hours),
                  identity_unit=a.identity_unit,
                  identity_scope=cache.get('speaker_id_scope', 'Historical speaker/session IDs'),
                  event_policy=f'{window}-score mean; rising latch;67-step refractory; reset between {a.identity_unit}s; all events counted',
                  exposure_policy='Original WAV seconds; exclude utterance gaps/alignment padding',
                  seconds=round(time.monotonic()-started, 1), speakers=results,
                  positive_recall_measured=False, development_regression_scored=a.split == "testing",
                  independent_final_scored=False,
                  limits=f'Development {a.corpus_name} negative audio only; Poisson event-count bound assumes stationary rate and does not establish general acoustic coverage, positive recall, deployment performance, or Hi ESP equivalence. Source-specific limits: {cache.get("limits", "Historical development source limits apply")}')
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
