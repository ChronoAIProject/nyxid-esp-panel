"""Separate input rounding from remaining INT8 error on all frozen witnesses."""
import argparse
import json
from pathlib import Path

import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT, published

PLAN = ROOT/'results/4090-hey-broad-fixed-grid-input-rounding-plan-v1.json'
OUT = Path('.local/wakeword/hey-broad-fixed-grid-input-rounding-v1')


def plan():
    original_path = ROOT/'results/4090-hey-broad-fixed-grid-screen-result-v1.json'
    original = json.loads(original_path.read_text())
    lock_path = ROOT/'results/4090-hey-broad-fixed-grid-screen-execution-v1.json'
    lock = json.loads(lock_path.read_text())
    assert original['execution_sha256'] == sha(lock_path)
    files = dict(lock['files'])
    files.update({str(p): sha(p) for p in (Path(__file__), original_path, lock_path)})
    for case in original['cases']:
        assert sha(case['arrays_path']) == case['arrays_sha256']
        files[case['arrays_path']] = case['arrays_sha256']
    result = dict(schema='hey-broad-fixed-grid-input-rounding-plan-v1', files=files,
        original_result_path=str(original_path), weights=lock['weights'], versions=lock['versions'],
        policy='All540 original complete training witnesses, same fixed-final H5. CPU2threads, explicit zero six states reset at whole-input start, actual saved quantized input dequantized with actual input scale/zero then cast float32. Twice-reset exact required. No optimizer, export, PTQ, thresholds or new development.',
        analysis='Compare original float F(x), float F(dequant(Q(x))) and saved final INT8 output on identical call clocks. Save all signed traces and five-mean mean/p99/max. Difference decomposition is descriptive, not independent additive causal effects through nonlinear recurrent state.',
        limits='Diagnostic only. Original failed gates unchanged; rounded-input reference cannot replace original full-precision reference for admission. No witness selection/retries or newdev/acceptance access.')
    with PLAN.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


def stats(delta):
    smoothed = np.asarray([np.mean(delta[i:i+5]) for i in range(len(delta)-4)])
    return dict(mean_absolute=float(np.mean(np.abs(smoothed))),
                p99_absolute=float(np.percentile(np.abs(smoothed), 99, method='linear')),
                maximum_absolute=float(np.max(np.abs(smoothed))))


def run():
    published(PLAN)
    locked = json.loads(PLAN.read_text())
    for path, expected in locked['files'].items():
        assert sha(path) == expected, path
    import importlib.metadata
    for name, version in locked['versions'].items():
        assert importlib.metadata.version(name) == version
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    tf = helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    original = json.loads(Path(locked['original_result_path']).read_text())
    cases = []
    from probe_hey_broad_bounded_float_v1 import bounded
    for source in locked['weights']:
        target=source['target'];condition=source['condition']
        model = helper.create(225); model.load_weights(source['path'])
        for layer in model.layers:
            cell = getattr(layer, 'cell', layer)
            if isinstance(cell, tf.keras.layers.BatchNormalization):
                layer.trainable = cell.trainable = False
        model=bounded(model,tf)
        stream = utils.to_streaming_inference(model, {'stride': 3, 'spectrogram_length': 225}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1, 3, 40), np.float32), training=False)
        states = [v for v in stream.variables if 'states' in v.name]; assert len(states) == 6
        infer = tf.function(lambda x: stream(x, training=False), input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
        for case in (c for c in original['cases'] if c['target'] == target and c['condition']==condition):
            with np.load(case['arrays_path'], allow_pickle=False) as saved:
                scale, zero = case['input_quantization']
                values = ((saved['quantized_input'].astype(np.float64)-zero)*scale).astype(np.float32)
                repeats = []
                for repeat in range(2):
                    for state in states:
                        state.assign(tf.zeros_like(state))
                    repeats.append(np.asarray([float(infer(values[None, at:at+3]).numpy()[0, 0]) for at in range(0, len(values), 3)]))
                assert np.array_equal(*repeats) and np.isfinite(repeats[0]).all()
                baseline = saved['float']; oscale, ozero = case['output_quantization']
                integer = (saved['uint8'].astype(np.float64)-ozero)*oscale
                rounded = repeats[0]
                assert len(baseline) == len(rounded) == len(integer)
                assert np.allclose((rounded-baseline)+(integer-rounded), integer-baseline, rtol=0, atol=1e-15)
                path = OUT/f'{target}-{condition}-{case["request_order"]:06}.npz'
                np.savez(path, rounded_input_float=rounded, repeat=repeats[1],
                         input_rounding_delta=rounded-baseline, remaining_delta=integer-rounded,
                         sample_end=saved['sample_end'])
                row = dict(target=target, condition=condition, request_order=case['request_order'], twice_reset_exact=True,
                    input_rounding=stats(rounded-baseline), remaining_quantization=stats(integer-rounded),
                    total_original=stats(integer-baseline), arrays_path=str(path), arrays_sha256=sha(path))
                cases.append(row)
                with (OUT/'cases.jsonl').open('a') as output:
                    output.write(json.dumps(row)+'\n')
                print(json.dumps(dict(target=target, cases=len(cases))), flush=True)
    assert len(cases)==540
    result = dict(schema='hey-broad-fixed-grid-input-rounding-result-v1', plan_sha256=sha(PLAN), cases=cases, limits=locked['limits'])
    with (ROOT/'results/4090-hey-broad-fixed-grid-input-rounding-result-v1.json').open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    args = parser.parse_args()
    plan() if args.action == 'plan' else run()
