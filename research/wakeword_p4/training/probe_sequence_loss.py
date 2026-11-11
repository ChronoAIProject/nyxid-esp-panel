"""Diagnose clipping/sigmoid saturation without training or scoring validation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

from sequence_loss import build_sequence_logits, mean_probability_bce
from sequence_scorer import build_sequence_scorer
from transfer import build_model


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compare(labels, values, mode):
    z = tf.constant(values, dtype=tf.float32)
    y = tf.constant(labels, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(z)
        probabilities = tf.math.sigmoid(z)
        reduced = probabilities[:, -1] if mode == 'terminal' else tf.reduce_mean(probabilities, axis=1)
        old = tf.keras.losses.binary_crossentropy(y[:, None], reduced[:, None])
    old_gradient = tape.gradient(old, z).numpy()
    with tf.GradientTape() as tape:
        tape.watch(z)
        stable = mean_probability_bce(y, z[:, -1:] if mode == 'terminal' else z)
    stable_gradient = tape.gradient(stable, z).numpy()
    arrays = {'old_loss': old.numpy(), 'stable_loss': stable.numpy(),
              'old_gradient': old_gradient, 'stable_gradient': stable_gradient}
    assert all(np.isfinite(a).all() for a in arrays.values())
    return arrays


def summarize(arrays):
    old_signal = np.max(np.abs(arrays['old_gradient']), axis=1)
    stable_signal = np.max(np.abs(arrays['stable_gradient']), axis=1)
    return {'rows': len(old_signal),
        'old_zero_gradient_stable_signal': int(np.count_nonzero((old_signal == 0) & (stable_signal > 1e-8))),
        'max_loss_difference': float(np.max(np.abs(arrays['old_loss']-arrays['stable_loss']))),
        'mean_absolute_loss_difference': float(np.mean(np.abs(arrays['old_loss']-arrays['stable_loss']))),
        'max_gradient_difference': float(np.max(np.abs(arrays['old_gradient']-arrays['stable_gradient'])))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'initial', 'study', 'upstream', 'plan', 'output'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_bytes())
    assert sha((args.study/'result.json').read_bytes()) == plan['source_result_sha256']
    for name, expected in plan['feature_metadata_sha256'].items():
        assert sha((args.features/name).read_bytes()) == expected
    draws_bytes = (args.study/'terminal-draws.jsonl').read_bytes()
    assert sha(draws_bytes) == plan['source_draws_sha256']
    draws = [json.loads(line) for line in draws_bytes.splitlines()][:plan['steps_selected']]
    assert len(draws) == 50
    sys.path.insert(0, str(args.upstream.resolve()))
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.features/'features.json').read_bytes())
    raw = np.memmap(args.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    metadata = {name: json.loads((args.features/name).read_bytes()) for name in ('ambient.json', 'human.json')}
    cached, data, labels, cohorts = {}, [], [], []
    for step, record in enumerate(draws, 1):
        assert record['step'] == step and len(record['draws']) == 16
        for at, (name, index, start, end) in enumerate(record['draws']):
            label = int(at < 4)
            if name == 'features':
                row = rows[index]
                assert row['split'] == 'training' and row['label'] == label and at < 10
                value = raw[row['offset']+start:row['offset']+end]
                engine = row['engine']
            else:
                row = metadata[name][index]
                assert row['split'] == 'training' and at >= 10
                key = (name, index)
                if key not in cached:
                    cached[key] = np.load(args.features/row['path'], mmap_mode='r', allow_pickle=False)
                value = cached[key][start:end]
                engine = name
            assert value.shape == (347, 40) and value.dtype == np.uint16
            data.append(value); labels.append(label); cohorts.append(engine+'/'+str(label))
    data = np.asarray(data, dtype='<u2')
    y = np.asarray(labels, dtype=np.float32)
    arrays = {'labels': y}
    scalar = []
    for case in plan['scalar_cases']:
        entry = dict(case)
        for mode in ('terminal', 'five_mean'):
            values = compare([case['label']], [case['logits']], mode)
            entry[mode] = {key: value.tolist() for key, value in values.items()}
        scalar.append(entry)
    summaries, sigmoid_errors = {}, {}
    for name, expected in plan['models'].items():
        weights = args.initial/'best.weights.h5' if name == 'initial' else args.study/(name+'.weights.h5')
        assert sha(weights.read_bytes()) == expected
        window, _ = build_model(335)
        window.load_weights(weights)
        before = [v.numpy().copy() for v in window.variables]
        logits_model = build_sequence_logits(window)
        probability_model = build_sequence_scorer(window)
        @tf.function(input_signature=[tf.TensorSpec([16, 347, 40], tf.float32)])
        def infer(x):
            return logits_model(x, training=False), probability_model(x, training=False)
        logits, probabilities = [], []
        for offset in range(0, len(data), 16):
            normalized = np.clip(data[offset:offset+16].astype(np.float32)/25.6, 0, 26)
            z, probability = infer(normalized)
            logits.append(z.numpy()); probabilities.append(probability.numpy())
        logits, probabilities = np.concatenate(logits), np.concatenate(probabilities)
        assert logits.shape == probabilities.shape == (800, 5)
        error = float(np.max(np.abs(tf.math.sigmoid(logits).numpy()-probabilities)))
        assert error < 1e-5
        sigmoid_errors[name] = error
        assert all(np.array_equal(v.numpy(), original) for v, original in zip(window.variables, before, strict=True))
        arrays[name+'_logits'] = logits
        arrays[name+'_probabilities'] = probabilities
        summaries[name] = {}
        for mode in ('terminal', 'five_mean'):
            values = compare(y, logits, mode)
            summaries[name][mode] = {'total': summarize(values), 'by_engine_label': {}}
            for cohort in sorted(set(cohorts)):
                mask = np.asarray(cohorts) == cohort
                summaries[name][mode]['by_engine_label'][cohort] = summarize({key: value[mask] for key, value in values.items()})
            for key, value in values.items():
                arrays[name+'_'+mode+'_'+key] = value
    np.savez_compressed(args.output/'arrays.npz', **arrays)
    (args.output/'plan.json').write_bytes(args.plan.read_bytes())
    report = {'scope': plan['scope'], 'plan_sha256': sha(args.plan.read_bytes()),
        'source_sha256': {name: sha((Path(__file__).parent/name).read_bytes()) for name in
                          ('probe_sequence_loss.py', 'sequence_loss.py', 'sequence_scorer.py', 'transfer.py')},
        'source_result_sha256': plan['source_result_sha256'], 'weights_sha256': plan['models'],
        'training_input_uint16_sha256': sha(data.tobytes()), 'labels_sha256': sha(y.tobytes()),
        'draw_rows': len(data), 'draw_cohorts': dict(Counter(cohorts)),
        'arrays_sha256': sha((args.output/'arrays.npz').read_bytes()),
        'sigmoid_logits_vs_original_probability_max_delta': sigmoid_errors,
        'all_model_variables_unchanged': True, 'optimizer_steps': 0,
        'scalar_cases': scalar, 'summary': summaries, 'tensorflow': tf.__version__,
        'limits': plan['limits'], 'candidate_promoted': False}
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'draw_rows': len(data), 'summary': {name: {mode: v['total'] for mode, v in values.items()}
                                                      for name, values in summaries.items()}}))


if __name__ == '__main__':
    main()
