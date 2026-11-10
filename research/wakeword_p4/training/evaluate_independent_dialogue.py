"""Score all frozen AMI dialogue streams at one separately preregistered point.

This evaluator never selects a threshold. It does not measure positive recall,
phonetic latency or hardware performance.
"""
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
        p.error('Fresh output required; preserve failed and completed final runs')
    freeze = json.loads(a.preregistration.read_text())
    manifest = json.loads(a.manifest.read_text())
    code = Path(__file__)
    if (freeze['final_role'] != 'independent-dialogue-final'
            or sha(a.model) != freeze['nyx_model_sha256']
            or sha(a.manifest) != freeze['pcm_manifest_sha256']
            or sha(code) != freeze['nyx_evaluator_source_sha256']
            or sha(code.with_name('evaluate.py')) != freeze['detector_source_sha256']
            or sha(code.with_name('features.py')) != freeze['frontend_source_sha256']
            or manifest['frontend_source_sha256'] != freeze['frontend_source_sha256']
            or manifest['source_audit_sha256'] != freeze['source_audit_sha256']
            or manifest['source_manifest_sha256'] != freeze['source_manifest_sha256']
            or not manifest['continuous_recordings'] or manifest['inserted_intermediate_gaps']
            or not manifest['original_pcm_preserved']):
        raise ValueError('Changed preregistered model, code, PCM population or source audit')
    rows = manifest['records']
    if (len(rows) != 16 or freeze['expected_streams'] != 16
            or len({r['session'] for r in rows}) != 16
            or len({r['path'] for r in rows}) != 16
            or len({r['feature_path'] for r in rows}) != 16
            or sorted(r['session'] for r in rows) != freeze['sessions']
            or any(r['split'] != 'testing' or r['role'] != freeze['final_role'] for r in rows)
            or len({x for r in rows for x in r['participants']}) != freeze['unique_participants']
            or sum(r['original_samples'] for r in rows)/16000/3600 != freeze['expected_source_hours']
            or freeze['confidence'] != .95 or freeze['false_per_hour_upper_95_limit'] != .5
            or freeze['sliding_window'] != 5 or not .5 <= freeze['nyx_threshold'] < 1):
        raise ValueError('Changed frozen test population, exposure or operating point')
    # Check every frozen input before loading a detector or creating output.
    for row in rows:
        for key, digest_key in [('path','sha256'),('feature_path','feature_sha256')]:
            path = (a.manifest.parent/row[key]).resolve()
            if not path.is_relative_to(a.manifest.parent.resolve()) or sha(path) != row[digest_key]:
                raise ValueError('Changed or escaped final input')
        data = np.load(a.manifest.parent/row['feature_path'],mmap_mode='r',allow_pickle=False)
        if (data.dtype != np.uint16 or data.shape != (row['frames'],40)
                or row['frames'] != row['samples']//160-2
                or row['samples'] != row['original_samples']+row['frontend_final_zero_pad_samples']
                or row['frontend_final_zero_pad_samples'] != (-row['original_samples']) % 160
                or row['source_seconds'] != row['original_samples']/16000):
            raise ValueError('Changed feature cadence, padding or original exposure')
    a.output.mkdir(parents=True)
    detector, results, failures = Detector(a.model), [], []
    started = time.monotonic()
    for row in rows:
        try:
            data = np.load(a.manifest.parent/row['feature_path'],mmap_mode='r',allow_pickle=False)
            detector.reset()
            prefix = detector.predict(data[:300])
            detector.reset()
            np.testing.assert_array_equal(prefix,detector.predict(data[:300]))
            detector.reset()
            scores = smooth(detector.predict(data),freeze['sliding_window'])
            if scores.shape != (len(data)//3-freeze['sliding_window']+1,) or not np.all(np.isfinite(scores)):
                raise ValueError('Incomplete or nonfinite score trace')
            indices = event_indices(scores,freeze['nyx_threshold'])
            path = a.output/(row['session']+'-scores.npy')
            np.save(path,scores)
            result = dict(session=row['session'],participants=row['participants'],
                          meeting_group=row['session'][:-1],source_seconds=row['source_seconds'],
                          pcm_sha256=row['pcm_sha256'],feature_sha256=row['feature_sha256'],
                          score_sha256=sha(path),prefix_reset_reproducible=True,
                          events=len(indices),event_steps=[int(i) for i in indices],
                          event_seconds=[float((3*(i+freeze['sliding_window']-1)+5)*.01) for i in indices])
            results.append(result)
            with (a.output/'progress.jsonl').open('a') as stream:
                stream.write(json.dumps(result)+'\n')
        except Exception as error:
            # Keep every failure and continue remaining sources; never summarize a subset.
            failures.append(dict(session=row['session'],error_type=type(error).__name__,error=str(error)))
        print(json.dumps(dict(completed=len(results),failures=len(failures),planned=16)),flush=True)
    if failures:
        (a.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        raise RuntimeError('Incomplete independent evaluation; no quality summary')

    def metrics(records):
        hours = sum(r['source_seconds'] for r in records)/3600
        events = sum(r['events'] for r in records)
        upper = float(.5*chi2.ppf(.95,2*(events+1))/hours)
        return dict(streams=len(records),events=events,source_hours=hours,
                    false_per_hour=events/hours,false_per_hour_upper_95=upper,
                    passes_numerical_bound=upper<=.5)

    report = dict(role='Independent AMI dialogue fixed-point negative test; not full-goal acceptance',
                  preregistration_sha256=sha(a.preregistration),manifest_sha256=sha(a.manifest),
                  source_sha256=sha(code),model_sha256=sha(a.model),threshold=freeze['nyx_threshold'],
                  sliding_window=freeze['sliding_window'],pooled=metrics(results),
                  by_meeting_group={g:metrics([r for r in results if r['meeting_group']==g])
                                    for g in sorted({r['meeting_group'] for r in results})},
                  records=results,seconds=round(time.monotonic()-started,1),
                  threshold_reselected=False,independent_dialogue_final_scored=True,
                  full_goal_achieved=False,
                  statistical_assumptions='Nominal Poisson one-sided95% bounds; repeated participants/conversation dependence and population shift are not covered.',
                  limits=freeze['limits'])
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['pooled']),flush=True)


if __name__ == '__main__':
    main()
