"""One fixed-point independent negative test under a committed preregistration."""
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
    for name in ('model', 'preregistration', 'manifest', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required; never overwrite a final run')
    freeze = json.loads(a.preregistration.read_text())
    manifest = json.loads(a.manifest.read_text())
    if (sha(a.model) != freeze['model_sha256'] or sha(a.manifest) != freeze['negative_manifest_sha256']
            or sha(Path(__file__).with_name('evaluate.py')) != freeze['detector_source_sha256']
            or sha(Path(__file__).with_name('features.py')) != freeze['frontend_source_sha256']
            or not manifest['development_speaker_disjoint']):
        raise ValueError('Model, manifest or scoring policy changed after preregistration')
    rows = manifest['records']
    if (len(rows) != freeze['expected_streams'] or len({r['speaker'] for r in rows}) != len(rows)
            or any(r['split'] != 'testing' or r['corpus'] not in ('test-clean', 'test-other') for r in rows)
            or abs(sum(r['source_seconds'] for r in rows)/3600-freeze['expected_source_hours']) > 1e-12):
        raise ValueError('Changed independent test population or exposure')
    # Validate every input before creating a run. Failed inference still preserves its directory.
    for row in rows:
        path = (a.manifest.parent/row['path']).resolve()
        if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row['feature_sha256']:
            raise ValueError('Changed or escaped frozen features')
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.dtype != np.uint16 or data.shape != (row['frames'], 40):
            raise ValueError('Invalid frozen feature shape/type')
    a.output.mkdir(parents=True)
    detector = Detector(a.model)
    results, started = [], time.monotonic()
    for row in rows:
        data = np.load(a.manifest.parent/row['path'], mmap_mode='r', allow_pickle=False)
        detector.reset()
        prefix = detector.predict(data[:300])
        detector.reset()
        np.testing.assert_array_equal(prefix, detector.predict(data[:300]))
        detector.reset()
        values = smooth(detector.predict(data), freeze['sliding_window'])
        indices = event_indices(values, freeze['threshold'])
        target = a.output/(str(row['speaker'])+'-scores.npy')
        np.save(target, values)
        result = dict(speaker=row['speaker'], corpus=row['corpus'], source_seconds=row['source_seconds'],
                      feature_sha256=row['feature_sha256'], score_sha256=sha(target),
                      events=len(indices), event_steps=[int(i) for i in indices],
                      event_seconds=[float((3*(i+freeze['sliding_window']-1)+5)*.01) for i in indices],
                      prefix_reset_reproducible=True)
        results.append(result)
        with (a.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
        print(json.dumps(dict(completed=len(results), planned=len(rows), events=len(indices))), flush=True)
    def metrics(rs):
        hours = sum(r['source_seconds'] for r in rs)/3600
        count = sum(r['events'] for r in rs)
        upper = float(.5*chi2.ppf(freeze['confidence'], 2*(count+1))/hours)
        return dict(streams=len(rs), events=count, source_hours=hours, false_per_hour=count/hours,
                    false_per_hour_upper_95=upper,
                    passes_numerical_bound=upper <= freeze['false_per_hour_upper_95_limit'])
    report = dict(role='Independent frozen negative fixed-point evaluation, not full goal acceptance',
                  preregistration_sha256=sha(a.preregistration), manifest_sha256=sha(a.manifest),
                  source_sha256=sha(Path(__file__)), model_sha256=sha(a.model),
                  detector_source_sha256=freeze['detector_source_sha256'],
                  threshold=freeze['threshold'], sliding_window=freeze['sliding_window'],
                  pooled=metrics(results), by_corpus={c:metrics([r for r in results if r['corpus'] == c])
                                                     for c in ('test-clean', 'test-other')},
                  speakers=results, seconds=round(time.monotonic()-started, 1),
                  threshold_reselected=False, independent_negative_final_scored=True,
                  independent_positive_final_scored=False, hiesp_final_measured=False,
                  full_goal_achieved=False, limits=freeze['limits'])
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['pooled']), flush=True)


if __name__ == '__main__':
    main()
