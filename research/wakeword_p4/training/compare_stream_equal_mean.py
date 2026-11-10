"""Compare fixed, equally weighted cached INT8 streams on reused validation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def checked(path, digest):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Changed frozen input: {path.name}')
    return data


def steps(values, threshold):
    high = np.flatnonzero(values >= threshold)
    low = np.flatnonzero(values < threshold * .5)
    accepted = []
    first = 0
    while first < len(high):
        last = int(high[first])
        accepted.append(last)
        rearm = np.searchsorted(low, last + 67)
        if rearm == len(low):
            break
        first = np.searchsorted(high, low[rearm])
    return accepted


def arm(root, name, binding, feature_index_hash):
    training, val = root / f'model-{name}', root / f'validation-{name}'
    checked(training / 'hey_nyx.tflite', binding['model_sha256'])
    meta = json.loads(checked(training / 'training.json', binding['training_sha256']))
    cal = json.loads(checked(val / 'threshold.json', binding['threshold_sha256']))
    if (meta['model_sha256'] != binding['model_sha256']
            or cal['model_sha256'] != binding['model_sha256']
            or cal['feature_index_sha256'] != feature_index_hash
            or cal.get('smoothing_steps', 5) != 5
            or cal['selection_scope'] != 'Validation only; threshold frozen before reading test/Mac audio'):
        raise ValueError('Original model, calibration or data identity differs')
    clips = json.loads(checked(val / 'validation-details.json', binding['validation_details_sha256']))
    streams = {}
    for filename, digest in binding['ambient_score_sha256'].items():
        checked(val / filename, digest)
        streams[filename] = np.load(val / filename, allow_pickle=False)
    return clips, streams


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifacts', 'plan', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report path')
    plan = json.loads(a.plan.read_bytes())
    if plan['event_policy'] != {'step_ms': 30, 'smoothing_steps': 5, 'refractory_steps': 67,
                               'rearm_fraction': .5, 'positive_proxy_window_seconds': [-.1, .8]}:
        raise ValueError('Unsupported event policy')
    name_a, name_b = plan['members']
    fa = plan['feature_index_sha256']
    base = a.artifacts / plan['validation_feature_directory']
    feature_rows = [r for r in json.loads(checked(base / 'features.json', fa)) if r['split'] == 'validation']
    ambient_rows = json.loads((base / 'human.json').read_bytes()) + json.loads((base / 'ambient.json').read_bytes())
    sides = [arm(a.artifacts, name, plan['members'][name], fa) for name in (name_a, name_b)]
    (details_a, ambient_a), (details_b, ambient_b) = sides
    if len(details_a['ambient']) != len(details_b['ambient']) or set(ambient_a) != set(ambient_b):
        raise ValueError('Different ambient coverage')
    hours = 0.
    expected_files = set()
    index = 0
    for i, row in enumerate(ambient_rows):
        if row['split'] != 'validation':
            continue
        key = f'ambient-validation-{i:03d}.npy'
        expected_files.add(key)
        frames = len(np.load(base / row['path'], allow_pickle=False, mmap_mode='r'))
        duration = frames * .01 / 3600
        if (ambient_a[key].shape != (frames // 3 - 4,) or ambient_b[key].shape != ambient_a[key].shape
                or any(d['ambient'][index]['hours'] != duration for d, _ in sides)):
            raise ValueError('Different continuous duration or sample alignment')
        hours += duration
        index += 1
    if set(ambient_a) != expected_files or index != len(details_a['ambient']):
        raise ValueError('Incomplete ambient index')

    records = []
    fields = ('engine', 'variant', 'label', 'text')
    if not len(details_a['validation']) == len(details_b['validation']) == len(feature_rows):
        raise ValueError('Different validation clip coverage')
    for ca, cb, row in zip(details_a['validation'], details_b['validation'], feature_rows, strict=True):
        if any(ca[k] != cb[k] or ca[k] != row[k] for k in fields) or not ca['source'] == cb['source'] == row['path']:
            raise ValueError('Different validation source')
        end = (row['keyword_frame'] + 2) * .01 if row['label'] else None
        if ca['keyword_end'] != end or cb['keyword_end'] != end or ca['times'] != cb['times']:
            raise ValueError('Different validation timebase')
        x, y = np.array(ca['scores']), np.array(cb['scores'])
        expected_times = (3 * (np.arange(row['frames'] // 3 - 4) + 4) + 5) * .01
        if x.shape != expected_times.shape or y.shape != x.shape or not np.array_equal(ca['times'], expected_times):
            raise ValueError('Invalid cached smoothing cadence')
        valid = (expected_times >= end - .1) & (expected_times <= end + .8) if end is not None else None
        records.append((row['engine'], row['path'].split('/')[0], valid, (x, y, (x + y) * .5)))
    streams = [(ambient_a[k], ambient_b[k], (ambient_a[k] + ambient_b[k]) * .5) for k in sorted(ambient_a)]
    strict = plan['strict']
    arms = {}
    for arm_index, name in enumerate((name_a, name_b, 'equal_mean')):
        grid = []
        for t in plan['grid']:
            positive, negative, grouped = defaultdict(list), defaultdict(list), defaultdict(list)
            for engine, corpus, valid, scores in records:
                values = scores[arm_index]
                if valid is not None:
                    positive[engine].append(any(valid[j] for j in steps(values, t)))
                else:
                    hit = bool(values.max() >= t)
                    negative[engine].append(hit)
                    grouped[engine, corpus].append(hit)
            macro = float(np.mean([np.mean(v) for v in positive.values()]))
            worst_engine = max(float(np.mean(v)) for v in negative.values())
            worst_group = max(float(np.mean(v)) for v in grouped.values())
            n = sum(len(steps(s[arm_index], t)) for s in streams)
            grid.append({'threshold': t, 'macro_latched_proxy_recall': macro,
                         'positive_clips_hit': sum(sum(v) for v in positive.values()),
                         'negative_clips_hit': sum(sum(v) for v in negative.values()),
                         'max_engine_false_clip_fraction': worst_engine,
                         'max_engine_corpus_false_clip_fraction': worst_group,
                         'continuous_events': n, 'continuous_events_per_hour': n / hours})
        def best(rows):
            return max(rows, key=lambda r: (r['macro_latched_proxy_recall'],
                       -r['continuous_events'], -r['max_engine_corpus_false_clip_fraction']), default=None)
        arms[name] = {'strict': best(r for r in grid
                                    if r['continuous_events_per_hour'] <= strict['continuous_events_per_hour_max']
                                    and r['max_engine_false_clip_fraction'] <= strict['max_engine_false_clip_fraction']
                                    and r['max_engine_corpus_false_clip_fraction'] <= strict['max_engine_corpus_false_clip_fraction']),
                      'zero_events_at_one_percent': best(r for r in grid if r['continuous_events'] == 0
                                                          and r['max_engine_corpus_false_clip_fraction'] <= .01),
                      'zero_events_at_two_percent': best(r for r in grid if r['continuous_events'] == 0
                                                          and r['max_engine_corpus_false_clip_fraction'] <= .02),
                      'grid': grid}
    candidate = arms['equal_mean']['strict']
    controls = [arms[k]['strict'] for k in (name_a, name_b) if arms[k]['strict']]
    if not controls:
        controls = [arms[k]['zero_events_at_two_percent'] for k in (name_a, name_b)
                    if arms[k]['zero_events_at_two_percent']]
    gain = candidate['macro_latched_proxy_recall'] - max(r['macro_latched_proxy_recall'] for r in controls) if (
        candidate and controls) else None
    result = {'scope': __doc__, 'plan_sha256': hashlib.sha256(a.plan.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'validation_clips': len(records), 'continuous_streams': len(streams), 'continuous_hours': hours,
              'macro_recall_gain_over_prespecified_control': gain, 'advance': gain is not None and gain >= .03,
              'limits': ['Identical cached validation timebases; no reserved evaluation, real-speaker claim or hardware.',
                         'Synthetic proxy endpoints are not phonetic timing measurements.',
                         'A qualifying result would motivate a single-model student, not deployment of two engines.'],
              'arms': arms}
    with a.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {p: v for p, v in r.items() if p != 'grid'} for k, r in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
