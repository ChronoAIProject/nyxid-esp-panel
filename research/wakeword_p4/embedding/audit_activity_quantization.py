"""Compare original float and INT8 heads at the already frozen INT8 thresholds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from audit_head_events import sha, trace_events
from evaluate_head import Head, summarize
from train_head import HISTORY, MIN_END, load_rows


def scores_float(predict, embeddings, smoothing):
    windows = np.lib.stride_tricks.sliding_window_view(embeddings, HISTORY, axis=0)
    windows = windows[MIN_END - HISTORY:].transpose(0, 2, 1)
    if len(windows) < smoothing:
        return np.empty(0, np.float64)
    raw = np.concatenate([predict(windows[start:start + 512, ..., None]).numpy().ravel()
                          for start in range(0, len(windows), 512)])
    if not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any():
        raise ValueError('Invalid float head scores')
    return np.convolve(raw, np.full(smoothing, 1 / smoothing), mode='valid')


def audit(arm, inputs, rows, thresholds):
    for filename, digest in inputs.items():
        if sha(arm / filename) != digest:
            raise ValueError(f'Original artifact differs: {filename}')
    validation = arm / 'validation'
    calibration = json.loads((validation / 'calibration.json').read_bytes())
    details = json.loads((validation / 'details-private.json').read_bytes())
    policy = calibration['event_policy']
    if (policy['refractory_steps'], policy['rearm_fraction'], policy['step_ms'],
            policy['score_average_steps'], policy['positive_window_seconds'], policy['min_history_frames']) != (
            25, .5, 80, 3, [-.1, .8], MIN_END):
        raise ValueError('Original event policy differs')
    clip_cache = {row['id']: row for row in details['clips']}
    continuous_cache = {row['id']: row for row in details['continuous']}
    if set(clip_cache) & set(continuous_cache) or len(rows) != len(clip_cache) + len(continuous_cache):
        raise ValueError('Original evaluation coverage differs')
    int8 = Head(arm / 'head.tflite')
    model = tf.keras.models.load_model(arm / 'head.keras', compile=False)

    @tf.function(input_signature=[tf.TensorSpec((None, HISTORY, 96, 1), tf.int8)])
    def predict(x):
        return model(tf.cast(x, tf.float32) / 128, training=False)

    score_hash = hashlib.sha256()
    point_counts = {threshold: {kind: Counter() for kind in ('positive', 'confuser', 'continuous')}
                    for threshold in thresholds}
    clips = {'float': [], 'int8': []}
    streams = {'float': [], 'int8': []}
    maxima, differences = {}, []
    for row in rows:
        name = row['id']
        quantized, times = int8.scores(row['array'], policy['score_average_steps'])
        floating = scores_float(predict, row['array'], policy['score_average_steps'])
        if len(floating) != len(times):
            raise ValueError(f'Window alignment differs: {name}')
        score_hash.update(name.encode() + b'\0')
        score_hash.update(floating.astype('<f8').tobytes())
        kind = 'positive' if row['label'] else 'confuser' if row['kind'] == 'tts' else 'continuous'
        if kind == 'continuous':
            old = continuous_cache.pop(name)
            np.testing.assert_array_equal(quantized, np.load(validation / (name + '-scores.npy'), allow_pickle=False))
            valid = np.ones(len(times), dtype=bool)
            evaluated = {'float': dict(old, events={}), 'int8': dict(old, events={})}
            for precision in evaluated:
                streams[precision].append(evaluated[precision])
        else:
            old = clip_cache.pop(name)
            if any(row[k] != old[k] for k in ('label', 'engine', 'corpus', 'variant', 'keyword_end')):
                raise ValueError(f'Original source metadata differs: {name}')
            np.testing.assert_array_equal(quantized, old['scores'])
            np.testing.assert_array_equal(times, old['times'])
            valid = ((times >= row['keyword_end'] - .1) & (times <= row['keyword_end'] + .8)
                     if kind == 'positive' else np.ones(len(times), dtype=bool))
            for precision, scores in (('float', floating), ('int8', quantized)):
                clips[precision].append(dict(old, scores=scores, max_score=float(scores[valid].max()) if valid.any() else 0.))
        error = np.abs(floating - quantized)
        differences.append(error)
        if len(error) and (kind not in maxima or error.max() > maxima[kind]['absolute_error']):
            at = int(error.argmax())
            maxima[kind] = {'id': name, 'time_seconds': float(times[at]), 'float': float(floating[at]),
                            'int8': float(quantized[at]), 'absolute_error': float(error[at])}
        for threshold in thresholds:
            a = trace_events(floating, threshold)
            b = trace_events(quantized, threshold)
            stats = point_counts[threshold][kind]
            stats.update({'streams': 1, 'float_events': len(a), 'int8_events': len(b),
                          'different_event_steps': int(a != b)})
            if kind == 'continuous':
                if len(b) != old['events'][str(threshold)]:
                    raise ValueError(f'Original continuous event count differs: {name}')
                evaluated['float']['events'][str(threshold)] = len(a)
                evaluated['int8']['events'][str(threshold)] = len(b)
            elif kind == 'positive':
                a, b = [at for at in a if valid[at]], [at for at in b if valid[at]]
            stats.update({'float_hits': int(bool(a)), 'int8_hits': int(bool(b)),
                          'float_only': int(bool(a) and not b), 'int8_only': int(bool(b) and not a)})
    if clip_cache or continuous_cache:
        raise ValueError('Original evaluation has unchecked rows')
    errors = np.concatenate(differences)
    report = {'float_score_sha256': score_hash.hexdigest(), 'smoothed_score_pairs': len(errors),
              'smoothed_mean_absolute_error': float(errors.mean()), 'smoothed_p99_absolute_error': float(np.quantile(errors, .99)),
              'maximum_discrepancies': maxima, 'original_int8_scores_reproduced_exactly': True, 'points': []}
    for threshold in thresholds:
        results = {p: summarize(clips[p], streams[p], threshold) for p in ('float', 'int8')}
        if results['int8'] != next(x for x in calibration['grid'] if x['threshold'] == threshold):
            raise ValueError('Original INT8 calibration summary differs')
        report['points'].append({'threshold': threshold, 'latched': point_counts[threshold], 'summary': results})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('experiment', 'data-root', 'plan', 'output'):
        p.add_argument('--' + arg, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh report path')
    plan = json.loads(a.plan.read_bytes())
    for name, digest in plan['scripts'].items():
        if sha(Path(__file__).parent / name) != digest:
            raise ValueError('Frozen helper source differs')
    for name, digest in plan['index_sha256'].items():
        if sha(a.data_root / name / 'index.jsonl') != digest:
            raise ValueError('Original validation index differs')
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    rows = load_rows('validation', a.data_root, plan['backbone_sha256'])
    report = {'scope': __doc__, 'plan_sha256': sha(a.plan), 'accepted': False,
              'speaker_playback': False, 'microphone_access': False, 'audio_speed_changed': False,
              'limits': ['Reused cached validation features; this checks detector decisions and synthetic proxy windows.',
                         'The original INT8 operating points are fixed; no float threshold selection, physical test or v30 claim.'],
              'arms': {}}
    for arm, files in plan['checkpoints'].items():
        report['arms'][arm] = audit(a.experiment / arm / 'step-10000', files, rows, plan['thresholds'])
        print(json.dumps({'completed': arm, 'score_pairs': report['arms'][arm]['smoothed_score_pairs']}), flush=True)
    with a.output.open('x') as f:
        f.write(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
