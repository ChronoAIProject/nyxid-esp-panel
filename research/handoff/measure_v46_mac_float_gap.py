"""Compare original v46 float export and saved INT8 outputs on all Mac350.

Both original float inputs and dequantized INT8 inputs are measured with the
same original SavedModel. No model export, fitting or threshold selection.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zlib

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import soundfile as sf
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'wakeword_p4/training'))
from features import features
from replay_gpu_mac_regression import groups, policy_result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory(variables):
    return Counter((tuple(v.shape), v.dtype.name, hashlib.sha256(v.numpy().tobytes()).hexdigest())
                   for v in variables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'model-dir', 'int8-study', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output required')
    plan = json.loads(args.plan.read_bytes())
    for path, pin in [(args.int8_study/'result.json', plan['int8_result_sha256']),
                      (args.int8_study/'scores.npz', plan['int8_scores_sha256']),
                      (args.model_dir/'training.json', plan['training_sha256']),
                      (args.model_dir/'hey_nyx.tflite', plan['tflite_sha256']),
                      (args.sources/'manifest.jsonl', plan['manifest_sha256']),
                      (args.sources/'baseline.json', plan['baseline_sha256'])]:
        require(sha(path) == pin, 'Original input differs: '+path.name)
    saved_path = args.model_dir/'saved_model'
    saved_files = {p.relative_to(saved_path).as_posix(): sha(p)
                   for p in sorted(saved_path.rglob('*')) if p.is_file()}
    require(saved_files == plan['saved_model_files'], 'Original SavedModel bytes differ')
    for package, pin in plan['versions'].items():
        require(version(package) == pin, 'Unexpected host version: '+package)
    prior = json.loads((args.int8_study/'result.json').read_bytes())
    traces = np.load(args.int8_study/'scores.npz', allow_pickle=False)
    manifest = [json.loads(line) for line in (args.sources/'manifest.jsonl').read_text().splitlines()]
    official = json.loads((args.sources/'baseline.json').read_bytes())
    require(len(manifest) == len(prior['source_details']) == len(official['clips']) == 350
            and set(traces.files) == {f'output_{i}' for i in range(350)}, 'Incomplete original study')
    # Authenticate all source files before loading the model or scoring a subset.
    for row, measured, old in zip(manifest, official['clips'], prior['source_details'], strict=True):
        require(all(measured[k] == old[k] == v for k, v in row.items()), 'Original source order differs')
        require(sha(args.sources/row['path']) == row['sha256'], 'Original source WAV differs')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    loaded = tf.saved_model.load(str(saved_path))
    signature = loaded.signatures['serving_default']
    inputs = signature.structured_input_signature[1]
    require(len(inputs) == 1 and tuple(next(iter(inputs.values())).shape) in ((1, 3, 40), (None, 3, 40)),
            'Unexpected original float-export input')
    key = next(iter(inputs))
    states = [v for v in loaded.variables if 'states' in v.name]
    nonstates = [v for v in loaded.variables if 'states' not in v.name]
    require(len(states) == 6 and len(nonstates) == 31, 'Unexpected original variable inventory')
    before = inventory(nonstates)

    def reset():
        for state in states:
            state.assign(tf.zeros_like(state))
        require(all(np.count_nonzero(state.numpy()) == 0 for state in states), 'Reset failed')

    @tf.function(input_signature=[tf.TensorSpec([None, 40], tf.float32)])
    def sequential_stream(values):
        count = tf.shape(values)[0]//3
        output = tf.TensorArray(tf.float32, size=count)

        def step(index, scores):
            value = signature(**{key: values[None, index*3:index*3+3]})
            return index+1, scores.write(index, tf.reshape(next(iter(value.values())), []))

        _, output = tf.while_loop(lambda index, _: index < count, step,
                                  (tf.constant(0), output), parallel_iterations=1)
        return output.stack()

    def replay(values):
        reset()
        output = sequential_stream(tf.convert_to_tensor(values)).numpy()
        require(output.dtype == np.float32 and np.isfinite(output).all()
                and np.all((output >= 0) & (output <= 1)), 'Invalid float probabilities')
        return output

    event_plan = dict(v30_threshold=plan['threshold'], smoothing_steps=5,
                      v30_block_samples=480, v30_prime_samples=56160)
    mode_rows = {mode: [] for mode in ('original', 'dequantized')}
    arrays, details, direct_checks = {}, [], []
    args.output.mkdir()
    shutil.copyfile(args.plan, args.output/'plan.json')
    shutil.copyfile(__file__, args.output/'source.py')
    started = time.monotonic()
    for index, (row, measured, old) in enumerate(zip(manifest, official['clips'], prior['source_details'], strict=True)):
        pcm, rate = sf.read(args.sources/row['path'], dtype='int16')
        require(rate == 16000 and pcm.ndim == 1 and len(pcm) == row['samples']
                and zlib.crc32(pcm.astype('<i2').tobytes()) == measured['result']['crc32'], 'PCM differs')
        tail = (-len(pcm)) % 480
        raw = features(np.pad(pcm, (56160, tail)).astype('<i2'))
        require(hashlib.sha256(raw.tobytes()).hexdigest() == old['frontend_sha256'], 'Frontend bytes differ')
        q = traces[f'output_{index}']
        require(q.dtype == np.uint8 and hashlib.sha256(q.tobytes()).hexdigest() == old['output_sha256'],
                'Saved INT8 output bytes differ')
        values = np.clip(raw.astype(np.float32)/25.6, 0, 26)
        scale, zero = plan['input_quantization']
        quantized = np.clip(np.rint(values/scale)+zero, -128, 127).astype(np.int8)
        dequantized = (quantized.astype(np.float32)-np.float32(zero))*np.float32(scale)
        ends = np.arange(6, len(q)+2)*480-56160
        int8_means = np.convolve(q.astype(np.float64)/256, np.ones(5)/5, mode='valid')
        record = {**row, 'frontend_sha256': old['frontend_sha256'], 'output_count': len(q),
                  'int8_max_post_prime_five_mean': float(int8_means[ends > 0].max()), 'modes': {}}
        for mode, data in [('original', values), ('dequantized', dequantized)]:
            scores = replay(data)
            require(len(scores) == len(q), 'Float/INT8 output geometry differs')
            np.testing.assert_array_equal(scores, replay(data))
            if index == 0:
                reset()
                direct = []
                for at in range(0, len(data)-2, 3):
                    output = signature(**{key: tf.convert_to_tensor(data[None, at:at+3])})
                    direct.append(float(next(iter(output.values())).numpy().reshape(-1)[0]))
                direct = np.asarray(direct, dtype=np.float32)
                np.testing.assert_allclose(scores, direct, atol=2e-5, rtol=2e-5)
                direct_checks.append(dict(mode=mode, compared=len(scores),
                    maximum_probability_error=float(np.max(np.abs(scores-direct)))))
            means = np.convolve(scores.astype(np.float64), np.ones(5)/5, mode='valid')
            output_ends = np.arange(2, len(scores)+2)*480-56160
            host, _ = policy_result(scores, event_plan)
            mode_rows[mode].append({**row, 'result': host})
            arrays[f'{mode}_{index}'] = scores
            record['modes'][mode] = dict(output_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                max_post_prime_score=float(scores[output_ends > 0].max()),
                max_post_prime_five_mean=float(means[ends > 0].max()),
                fixed_threshold_events=host, reset_replay_exact=True)
        details.append(record)
        if index % 25 == 0:
            print(json.dumps(dict(completed=index+1, total=350,
                                 seconds=round(time.monotonic()-started, 2))), flush=True)
    require(inventory(nonstates) == before, 'Original nonstate weights changed')
    np.savez_compressed(args.output/'float-scores.npz', **arrays)
    zeros = [row for row in details if row['target'] == 'hey_nyx'
             and row['int8_max_post_prime_five_mean'] == 0]
    result = dict(scope=__doc__, accepted=False, plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)),
                  saved_model_files=saved_files, versions=plan['versions'], sources=350,
                  quantized_reference_outputs=sum(row['output_count'] for row in details),
                  float_outputs=sum(len(value) for value in arrays.values()),
                  float_scores_sha256=sha(args.output/'float-scores.npz'),
                  int8_result_sha256=plan['int8_result_sha256'], int8_scores_sha256=plan['int8_scores_sha256'],
                  states_reset_per_stream=6, nonstate_variables_unchanged=True,
                  all_mode_reset_replays_exact=True, first_source_direct_signature_checks=direct_checks,
                  threshold=plan['threshold'], summaries={mode: groups(rows, 'hey_nyx') for mode, rows in mode_rows.items()},
                  original_int8_zero_targets=len(zeros), zero_target_details=zeros,
                  source_details=details, limits=plan['limits'])
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(sources=350, zero_targets=len(zeros),
                          summaries={mode: value['all'] for mode, value in result['summaries'].items()})))


if __name__ == '__main__':
    main()
