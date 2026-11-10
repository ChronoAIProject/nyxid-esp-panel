"""Trace existing calibrated validation scores without changing model or windows.

Counts actual latched threshold events separately from the legacy peak metric.
This is score-policy replay, not microphone/P4 execution or phonetic timing.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_events(scores, threshold):
    values = np.asarray(scores, np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any()
            or (values > 1).any() or not 0 < threshold < 1):
        raise ValueError('Invalid probability stream or threshold')
    armed, last, result = True, -1000, []
    for index, value in enumerate(values):
        if value < threshold * .5 and index - last >= 25:
            armed = True
        if armed and value >= threshold and index - last >= 25:
            result.append(index)
            last, armed = index, False
    return result


def audit(checkpoint):
    calibration = json.loads((checkpoint / 'validation/calibration.json').read_text())
    details_path = checkpoint / 'validation/details-private.json'
    details = json.loads(details_path.read_text())
    threshold = calibration['selected']['threshold']
    policy = calibration['event_policy']
    if (sha(checkpoint / 'head.tflite') != calibration['model_sha256']
            or policy['refractory_steps'] != 25 or policy['rearm_fraction'] != .5
            or policy['step_ms'] != 80 or policy['score_average_steps'] != 3
            or policy['positive_window_seconds'] != [-.1, .8]):
        raise ValueError('Frozen model/event/legacy-window identity differs')
    records = []
    for clip in details['clips']:
        scores, times = np.asarray(clip['scores']), np.asarray(clip['times'])
        if len(scores) != len(times) or not np.allclose(np.diff(times), .08, atol=1e-12, rtol=0):
            raise ValueError('Score/time coverage differs')
        indices = trace_events(scores, threshold)
        event_times = times[indices]
        valid = ((times >= clip['keyword_end'] - .1) & (times <= clip['keyword_end'] + .8)) if clip['label'] else np.ones(len(scores), bool)
        maximum = float(scores[valid].max()) if valid.any() else 0.
        if maximum != clip['max_score']:
            raise ValueError('Legacy clip peak does not reproduce')
        record = {key: clip[key] for key in ('id', 'label', 'engine', 'corpus', 'variant', 'keyword_end')}
        record.update(event_times_seconds=event_times.tolist(), native_event_count=len(indices),
                      original_peak_hit=clip['max_score'] >= threshold)
        if clip['label']:
            end = clip['keyword_end']
            within = (event_times >= end - .1) & (event_times <= end + .8)
            record.update(native_event_in_original_proxy_window=bool(within.any()),
                native_event_before_original_proxy_window=bool((event_times < end - .1).any()),
                peak_hit_without_native_event_in_proxy_window=bool(record['original_peak_hit'] and not within.any()))
        records.append(record)
    continuous = []
    for source in details['continuous']:
        path = checkpoint / 'validation' / (source['id'] + '-scores.npy')
        scores = np.load(path, allow_pickle=False)
        indices = trace_events(scores, threshold)
        if len(indices) != source['events'][str(threshold)]:
            raise ValueError('Native trace differs from existing continuous event accounting')
        continuous.append({'id': source['id'], 'scores_sha256': sha(path),
            'events': len(indices), 'event_indices': indices, 'evaluated_hours': source['evaluated_hours']})
    groups = defaultdict(list)
    for record in records:
        groups[(record['label'], record['engine'])].append(record)
    summary = []
    for (label, engine), rows in sorted(groups.items()):
        counts = {'label': label, 'engine': engine, 'variants': len(rows),
            'original_peak_hits': sum(r['original_peak_hit'] for r in rows),
            'native_event_sources_anywhere': sum(r['native_event_count'] > 0 for r in rows),
            'native_events': sum(r['native_event_count'] for r in rows)}
        if label:
            for key in ('native_event_in_original_proxy_window', 'native_event_before_original_proxy_window',
                        'peak_hit_without_native_event_in_proxy_window'):
                counts[key] = sum(r[key] for r in rows)
        summary.append(counts)
    return {'scope': __doc__, 'script_sha256': sha(Path(__file__)),
        'model_sha256': calibration['model_sha256'], 'calibration_sha256': sha(checkpoint / 'validation/calibration.json'),
        'details_sha256': sha(details_path), 'threshold': threshold, 'event_policy': policy,
        'threshold_or_window_reselected': False, 'groups': summary, 'records': records,
        'continuous': continuous, 'continuous_events': sum(r['events'] for r in continuous),
        'continuous_evaluated_hours': sum(r['evaluated_hours'] for r in continuous),
        'limits': ['Latched-policy replay uses already calibrated reused validation scores, not P4 or acoustic execution.',
            'Any event is unfiltered and may precede the intended word; original proxy-window membership is kept separately.',
            'Do not promote any-event or proxy counts to real-user recall or phonetic latency.',
            'No new threshold, labels, source windows, audio or runtime behavior is fitted.'], 'accepted': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    with args.output.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'groups': result['groups'], 'continuous_events': result['continuous_events'], 'accepted': False}))


if __name__ == '__main__':
    main()
