"""Registered float preactivation observation; no training, PTQ or export."""
import argparse
import json
from pathlib import Path

import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT, published

PLAN = ROOT/'results/4090-hey-broad-logit-sentinel-plan-v1.json'
OUT = Path('.local/wakeword/hey-broad-logit-sentinel-v1')


def plan():
    source = ROOT/'results/4090-hey-broad-input-rounding-plan-v1.json'
    prior = json.loads(source.read_text())
    files = dict(prior['files'])
    files.update({str(source): sha(source), str(Path(__file__)): sha(__file__)})
    result = dict(schema='hey-broad-logit-sentinel-plan-v1', files=files,
        versions=prior['versions'], models=prior['models'], original_result_path=prior['original_result_path'],
        arms=['natural', 'zero_zero', 'zero_full26'], reset_repeats=2,
        policy='All270 original training witness arrays. CPU2threads same original final H5; observational stream outputs probability and original Dense input in one shared graph call. Compute original Dense matmul+bias as preactivation. Verify sigmoid(logit) matches probability atol1e-5, natural probability matches saved original float atol1e-4. Fresh six-state reset twice per complete arm. Natural has no prelude; other arms have exactly two3x40feature inputs, zero+zero versus zero+26, followed by unchanged entire original feature array with final-only zero padding. No timing/gate reinterpretation.',
        hypothesis='Original representative zero/full26 sentinel sequence may produce extreme logits and pollute later state ranges. Compare fixed prefixes and full recovery trajectories, not only the two sentinel calls.',
        limits='Diagnostic feature-domain prefix intervention, not real audio, PTQ range replay or a candidate. Does not replay original100 PTQ samples or identify converter calibration aggregation. No optimization/export/PTQ retry, newdev, acceptance, threshold/tolerance change.')
    with PLAN.open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


def run():
    published(PLAN); locked = json.loads(PLAN.read_text())
    for path, expected in locked['files'].items():
        assert sha(path) == expected, path
    import importlib.metadata
    for name, expected in locked['versions'].items():
        assert importlib.metadata.version(name) == expected
    OUT.mkdir(exist_ok=False)
    import probe_hey_context_transfer_shapes_v1 as helper
    tf = helper.tf
    from microwakeword import utils
    from microwakeword.layers import modes
    original = json.loads(Path(locked['original_result_path']).read_text())
    cases = []
    for target, folder in locked['models'].items():
        model = helper.create(225); model.load_weights(Path(folder)/'best.weights.h5')
        for layer in model.layers:
            cell = getattr(layer, 'cell', layer)
            if isinstance(cell, tf.keras.layers.BatchNormalization):
                layer.trainable = cell.trainable = False
        stream = utils.to_streaming_inference(model, {'stride': 3, 'spectrogram_length': 225}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1, 3, 40), np.float32), training=False)
        states = [v for v in stream.variables if 'states' in v.name]; assert len(states) == 6
        heads = [layer for layer in stream.layers if isinstance(layer, tf.keras.layers.Dense)]
        assert len(heads) == 1
        head = heads[0]
        observer = tf.keras.Model(inputs=stream.inputs, outputs=[stream.outputs[0], head.input])
        @tf.function(input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
        def infer(value):
            probability, embedding = observer(value, training=False)
            return probability, tf.linalg.matmul(embedding, head.kernel)+head.bias
        for case in (c for c in original['cases'] if c['target'] == target):
            with np.load(case['arrays_path'], allow_pickle=False) as saved:
                raw = saved['raw']; baseline = saved['float']
            values = np.clip(np.pad(raw, ((0, (-len(raw)) % 3), (0, 0))).astype(np.float32)/25.6, 0, 26)
            arms = {}
            for arm in locked['arms']:
                prefix = [] if arm == 'natural' else [np.zeros((1, 3, 40), np.float32),
                    np.full((1, 3, 40), 26 if arm == 'zero_full26' else 0, np.float32)]
                repeats = []
                for repeat in range(locked['reset_repeats']):
                    for state in states:
                        state.assign(tf.zeros_like(state))
                    output = []
                    for value in prefix + [values[None, at:at+3] for at in range(0, len(values), 3)]:
                        probability, logit = infer(value)
                        output.append((float(probability.numpy()[0, 0]), float(logit.numpy()[0, 0])))
                    repeats.append(np.asarray(output))
                assert np.array_equal(*repeats) and np.isfinite(repeats[0]).all()
                probability, logits = repeats[0].T
                # Stable independent sigmoid formula.
                reconstructed = np.exp(-np.logaddexp(0, -logits))
                error = float(np.max(np.abs(probability-reconstructed))); assert error <= 1e-5
                if arm == 'natural':
                    assert np.max(np.abs(probability-baseline)) <= 1e-4
                path = OUT/f'{target}-{case["request_order"]:06}-{arm}.npz'
                np.savez(path, probability=probability, logit=logits, repeat=repeats[1])
                real_logits = logits[len(prefix):]
                arms[arm] = dict(logit_min=float(logits.min()), logit_max=float(logits.max()),
                    after_prefix_min=float(real_logits.min()), after_prefix_max=float(real_logits.max()),
                    first_two_logits=logits[:2].tolist(), sigmoid_reconstruction_max_error=error,
                    twice_reset_exact=True, arrays_path=str(path), arrays_sha256=sha(path))
            row = dict(target=target, request_order=case['request_order'], arms=arms)
            cases.append(row)
            with (OUT/'cases.jsonl').open('a') as output:
                output.write(json.dumps(row)+'\n')
            print(json.dumps(dict(cases=len(cases), target=target)), flush=True)
    result = dict(schema='hey-broad-logit-sentinel-result-v1', plan_sha256=sha(PLAN), cases=cases, limits=locked['limits'])
    with (ROOT/'results/4090-hey-broad-logit-sentinel-result-v1.json').open('x') as output:
        output.write(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    args = parser.parse_args()
    plan() if args.action == 'plan' else run()
