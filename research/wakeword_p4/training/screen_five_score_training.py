"""Matched terminal/five-score short training screen on frozen old features."""
import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

from sequence_scorer import build_sequence_scorer
from transfer import build_model


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2)+'\n').encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'initial', 'upstream', 'plan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    plan = json.loads(a.plan.read_bytes())
    for name, digest in plan['feature_metadata_sha256'].items():
        assert sha((a.features/name).read_bytes()) == digest
    assert sha((a.initial/'best.weights.h5').read_bytes()) == plan['initial_weights_sha256']
    assert sha((a.initial/'training.json').read_bytes()) == plan['initial_training_sha256']
    upstream = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, text=True).strip()
    assert upstream == plan['upstream_revision']
    sys.path.insert(0, str(a.upstream.resolve()))
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(plan['seed'])
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'plan.json').write_bytes(a.plan.read_bytes())
    frames, span = plan['window_frames'], plan['sequence_crop_frames']
    assert span == frames+12 and frames == 335
    rows = json.loads((a.features/'features.json').read_bytes())
    raw = np.memmap(a.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    groups = {label: defaultdict(list) for label in (0, 1)}
    excluded = Counter()
    for i, row in enumerate(rows):
        if row['split'] != 'training':
            continue
        eligible = row['frames'] >= span
        if row['label']:
            eligible = eligible and row['keyword_frame']+15 >= span and row['keyword_frame']+15 <= row['frames']
        if eligible:
            groups[row['label']][row['engine']].append(i)
        else:
            excluded[f"{row['engine']}/{row['label']}"] += 1
    ambient = []
    for name in ('human.json', 'ambient.json'):
        for i, row in enumerate(json.loads((a.features/name).read_bytes())):
            if row['split'] == 'training':
                # No testing/validation ambient array is opened.
                data = np.load(a.features/row['path'], mmap_mode='r', allow_pickle=False)
                assert data.ndim == 2 and data.shape[1] == 40 and data.dtype == np.uint16
                if len(data) >= span:
                    ambient.append((name, i, data))
    assert ambient and all(groups.values())

    def source_data(index, split):
        row = rows[index]
        assert row['split'] == split
        return raw[row['offset']:row['offset']+row['frames']]

    def batch(rng):
        values, labels, draws = [], [], []
        for label, count in ((1, 4), (0, 6)):
            for _ in range(count):
                engine = str(rng.choice(sorted(groups[label])))
                index = int(rng.choice(groups[label][engine]))
                row = rows[index]
                data = source_data(index, 'training')
                end = (row['keyword_frame']+int(rng.integers(15, min(27, len(data)-row['keyword_frame'])+1))
                       if label else int(rng.integers(span, len(data)+1)))
                values.append(data[end-span:end]); labels.append(label)
                draws.append(['features', index, end-span, end])
        for _ in range(6):
            name, index, data = ambient[int(rng.integers(len(ambient)))]
            start = int(rng.integers(0, len(data)-span+1))
            values.append(data[start:start+span]); labels.append(0)
            draws.append([name, index, start, start+span])
        array, labels = np.asarray(values, dtype=np.uint16), np.asarray(labels, dtype=np.float32)
        assert array.shape == (16, span, 40)
        return array, labels, draws

    def make_model():
        window, _ = build_model(frames)
        window.load_weights(a.initial/'best.weights.h5')
        assert window.count_params() == 25831
        return window, build_sequence_scorer(window)

    def normalized(data):
        return np.clip(data.astype(np.float32)/25.6, 0, 26)

    # Validate the mapping with the actual learned warm start before fitting.
    window, sequence = make_model()
    first, _, _ = batch(np.random.default_rng(plan['seed']))
    x = normalized(first)
    predicted = sequence(x, training=False).numpy()
    reference = np.stack([window(x[:, at:at+frames], training=False).numpy()[:, 0]
                          for at in range(0, 15, 3)], axis=1)
    assert predicted.shape == reference.shape == (16, 5)
    preflight_delta = float(np.max(np.abs(predicted-reference)))
    (a.output/'preflight.json').write_bytes(encode({'max_delta': preflight_delta}))
    np.testing.assert_allclose(predicted, reference, rtol=0, atol=1e-4)

    selected = []
    for engine, label in sorted({(r['engine'], r['label']) for r in rows if r['split'] == 'validation'}):
        group = [(i, r) for i, r in enumerate(rows)
                 if r['split'] == 'validation' and (r['engine'], r['label']) == (engine, label)]
        keys = set(sorted({r['sha256'] for _, r in group})[:16])
        selected.extend(i for i, r in group if r['sha256'] in keys and r['variant'] in (0, 3))
    assert selected and all(rows[i]['split'] == 'validation' for i in selected)
    (a.output/'validation-selection.json').write_bytes(encode([{'index': i, **rows[i]} for i in selected]))

    def evaluate(scorer, name):
        @tf.function(input_signature=[tf.TensorSpec([1, None, 40], tf.float32)])
        def infer(data):
            return scorer(data, training=False)
        details, arrays = [], {}
        for index in selected:
            row = rows[index]
            data = source_data(index, 'validation')
            scores = infer(normalized(data[1:])[None]).numpy()[0]
            assert np.isfinite(scores).all()
            means = np.convolve(scores, np.ones(5)/5, mode='valid')
            armed, last, steps = True, -1000, []
            # First mature score is one-based raw score112, then convolution
            # consumes another4. Retain returning 480-sample block offsets.
            for step, value in enumerate(means, 116):
                if value < plan['threshold']/2 and step-last >= 67:
                    armed = True
                if armed and value >= plan['threshold'] and step-last >= 67:
                    armed, last = False, step; steps.append(step)
            offsets = [480*(s+1) for s in steps]
            end = (row['keyword_frame']+2)*160 if row['label'] else None
            detail = {'index': index, 'engine': row['engine'], 'label': row['label'],
                'path': row['path'], 'sha256': row['sha256'], 'variant': row['variant'],
                'input_feature_sha256': sha(data.astype('<u2').tobytes()),
                'scores': len(scores), 'score_sha256': sha(scores.tobytes()),
                'mature_event_steps': steps, 'return_offsets': offsets, 'any_event': bool(steps),
                'estimated_end_window_hit': any(end-1600 <= at <= end+12800 for at in offsets) if end is not None else None}
            details.append(detail); arrays[f'source_{index}'] = scores
        np.savez_compressed(a.output/(name+'-scores.npz'), **arrays)
        summary = {}
        for engine in sorted({r['engine'] for r in details}):
            group = [r for r in details if r['engine'] == engine]
            positives, negatives = [r for r in group if r['label']], [r for r in group if not r['label']]
            summary[engine] = {'positive_clips': len(positives), 'negative_clips': len(negatives),
                'positive_any_hits': sum(r['any_event'] for r in positives),
                'positive_estimated_window_hits': sum(r['estimated_end_window_hit'] for r in positives),
                'negative_false_clips': sum(r['any_event'] for r in negatives)}
        result = {'summary': summary, 'details': details,
                  'scores_sha256': sha((a.output/(name+'-scores.npz')).read_bytes())}
        (a.output/(name+'-evaluation.json')).write_bytes(encode(result))
        print(json.dumps({'evaluation': name, 'summary': summary}), flush=True)
        return result

    initial_eval = None if a.smoke else evaluate(sequence, 'initial')
    arms = {}
    for arm in ('terminal', 'five_mean'):
        window, sequence = make_model()
        initial_values = [v.numpy().copy() for v in window.variables]
        optimizer = tf.keras.optimizers.Adam(plan['learning_rate'], global_clipnorm=plan['gradient_clipnorm'])

        @tf.function(input_signature=[tf.TensorSpec([16, span, 40], tf.float32), tf.TensorSpec([16], tf.float32)])
        def fit(data, labels):
            with tf.GradientTape() as tape:
                scores = sequence(data, training=True)
                probability = scores[:, -1] if arm == 'terminal' else tf.reduce_mean(scores, axis=1)
                losses = tf.keras.losses.binary_crossentropy(labels[:, None], probability[:, None])
                loss = tf.reduce_mean(losses*tf.where(labels > 0, 1., 20.))
            gradients = tape.gradient(loss, window.trainable_variables)
            for gradient in gradients:
                tf.debugging.assert_all_finite(gradient, 'Invalid training gradient')
            optimizer.apply_gradients(zip(gradients, window.trainable_variables))
            return loss

        rng, digest, started = np.random.default_rng(plan['seed']), hashlib.sha256(), time.monotonic()
        count = 2 if a.smoke else plan['steps']
        losses = []
        with (a.output/(arm+'-draws.jsonl')).open('x') as draws_file:
            for step in range(1, count+1):
                data, labels, draws = batch(rng)
                digest.update(data.astype('<u2').tobytes()); digest.update(labels.astype('<f4').tobytes())
                draws_file.write(json.dumps({'step': step, 'draws': draws})+'\n')
                loss = float(fit(normalized(data), labels).numpy())
                assert np.isfinite(loss)
                losses.append(loss)
                if step % 50 == 0 or step == count:
                    print(json.dumps({'arm': arm, 'step': step, 'loss': loss,
                                      'elapsed_seconds': round(time.monotonic()-started, 2)}), flush=True)
        assert all(np.isfinite(v.numpy()).all() for v in window.variables)
        differences = [v.numpy()-initial for v, initial in zip(window.variables, initial_values, strict=True)]
        change = float(sum(np.sum(d.astype(np.float64)**2) for d in differences)**.5)
        assert change > 0
        window.save_weights(a.output/(arm+'.weights.h5'))
        arms[arm] = {'steps': count, 'batch_bytes_sha256': digest.hexdigest(),
            'weights_sha256': sha((a.output/(arm+'.weights.h5')).read_bytes()),
            'draws_sha256': sha((a.output/(arm+'-draws.jsonl')).read_bytes()),
            'parameter_l2_change': change,
            'seconds': round(time.monotonic()-started, 2), 'first_loss': losses[0], 'last_loss': losses[-1],
            'evaluation': None if a.smoke else evaluate(sequence, arm)}
    assert arms['terminal']['batch_bytes_sha256'] == arms['five_mean']['batch_bytes_sha256']
    assert arms['terminal']['draws_sha256'] == arms['five_mean']['draws_sha256']
    if not a.smoke:
        inputs = [[(r['index'], r['input_feature_sha256']) for r in result['details']]
                  for result in (initial_eval, arms['terminal']['evaluation'], arms['five_mean']['evaluation'])]
        assert inputs[0] == inputs[1] == inputs[2]
    result = {'scope': plan['scope'], 'smoke': a.smoke, 'plan_sha256': sha(a.plan.read_bytes()),
        'source_sha256': sha(Path(__file__).read_bytes()),
        'sequence_scorer_sha256': sha(Path(__file__).with_name('sequence_scorer.py').read_bytes()),
        'initial_weights_sha256': plan['initial_weights_sha256'], 'preflight_max_score_delta': preflight_delta,
        'excluded_training_rows': dict(excluded), 'validation_selected_rows': len(selected),
        'versions': {name: version(name) for name in ('tensorflow', 'numpy', 'h5py')},
        'initial_evaluation': initial_eval, 'arms': arms, 'limits': plan['limits'],
        'test_rows_accessed': 0, 'candidate_accepted_or_promoted': False}
    (a.output/'result.json').write_bytes(encode(result))
    print(json.dumps({'done': True, 'smoke': a.smoke, 'same_batches': True}), flush=True)


if __name__ == '__main__':
    main()
