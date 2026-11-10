"""Frozen export-only lexical gate; reuse weights without fitting any new data.

One shared trunk feeds the existing binary sigmoid and existing six-class
classifier. Output is binary_probability * lexical_target_probability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args):
    import tensorflow as tf
    from train_head import export

    plan = json.loads(args.plan.read_text())
    for name, expected in plan['scripts'].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError('Frozen composition/helper source differs')
    for name, expected in plan['source_files'].items():
        if sha(args.source / name) != expected:
            raise ValueError('Frozen trained source artifact differs')
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    started = time.monotonic()
    original = tf.keras.models.load_model(args.source / 'step-10000/head.keras')
    auxiliary = tf.keras.models.load_model(args.source / 'training-only-auxiliary.keras')
    common = {layer.name: layer for layer in original.layers[:-1]}
    for layer in auxiliary.layers[:-1]:
        if (layer.name not in common or type(layer) is not type(common[layer.name])
                or len(layer.get_weights()) != len(common[layer.name].get_weights())
                or any(not np.array_equal(a, b) for a, b in zip(layer.get_weights(), common[layer.name].get_weights()))):
            raise ValueError('Source trunk weights are not shared exactly')
    if (set(common) != {layer.name for layer in auxiliary.layers[:-1]}
            or auxiliary.layers[-1].units != 6 or original.output.shape[-1] != 1):
        raise ValueError('Unexpected binary/lexical source topology')
    classifier = tf.keras.layers.Dense(6, name='lexical_logits', kernel_initializer='zeros')
    logits = classifier(original.layers[-2].output)
    classifier.set_weights(auxiliary.layers[-1].get_weights())
    posterior = tf.keras.layers.Softmax(name='lexical_posterior')(logits)
    target = posterior[:, :1]
    product = tf.keras.layers.Multiply(name='gated_wake_probability')([original.output, target])
    model = tf.keras.Model(original.input, product)
    if model.count_params() != original.count_params() + auxiliary.layers[-1].count_params():
        raise ValueError('Composed graph duplicates the shared trunk')
    examples = np.load(args.source / 'representatives-training-only.npy', allow_pickle=False)
    expected = np.asarray(original(examples, training=False)) * np.asarray(tf.nn.softmax(auxiliary(examples, training=False)))[:, :1]
    actual = np.asarray(model(examples, training=False))
    error = float(np.max(np.abs(expected - actual)))
    if error > 1e-6 or np.any(actual > np.asarray(original(examples, training=False)) + 1e-6):
        raise ValueError('Float composition does not reproduce the frozen source product')
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / 'step-10000'
    checkpoint.mkdir()
    np.save(args.output / 'representatives-training-only.npy', examples)
    report = export(model, checkpoint, examples)
    source_complete = json.loads((args.source / 'complete.json').read_text())
    result = {**source_complete, 'scope': __doc__, 'plan': plan,
        'plan_sha256': sha(args.plan), 'source_files': plan['source_files'],
        'source_training_complete_sha256': sha(args.source / 'complete.json'),
        'parameters': model.count_params(), 'new_training_steps': 0,
        'auxiliary_parameters_training_only': 0,
        'lexical_parameters_retained_at_inference': classifier.count_params(),
        'all_source_trunk_weights_exact': True, 'trunk_duplicated': False,
        'composition_max_float_error': error, 'checkpoints': [{'step': 10000, **report}],
        'seconds': time.monotonic() - started, 'accepted': False}
    (args.output / 'complete.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('parameters', 'new_training_steps', 'composition_max_float_error', 'checkpoints', 'seconds')}), flush=True)


def compare(args):
    from compare_lexical_auxiliary import confidence_point
    sides = {}
    for name, root in [('control', args.source), ('lexical_product', args.output)]:
        complete = json.loads((root / 'complete.json').read_text())
        calibration_path = root / 'step-10000/validation/calibration.json'
        calibration = json.loads(calibration_path.read_text())
        if (sha(root / 'step-10000/head.tflite') != calibration['model_sha256']
                or complete['checkpoints'][0]['model_sha256'] != calibration['model_sha256']
                or calibration['event_policy'] != complete['plan']['event_policy']):
            raise ValueError('Composed/source model or validation identity differs')
        sides[name] = {'complete_sha256': sha(root / 'complete.json'), 'calibration_sha256': sha(calibration_path),
            'details_sha256': sha(root / 'step-10000/validation/details-private.json'),
            'export': complete['checkpoints'][0], 'parameters': complete['parameters'],
            'original_selected': calibration['selected'], 'strict_selection': calibration['selection'],
            'confidence_and_confuser_qualified': confidence_point(calibration['grid']),
            'confidence_only_diagnostic': confidence_point(calibration['grid'], False), 'full_grid': calibration['grid']}
    left = json.loads((args.source / 'complete.json').read_text())
    right = json.loads((args.output / 'complete.json').read_text())
    if (left['feature_inputs'] != right['feature_inputs']
            or sha(args.source / 'representatives-training-only.npy') != sha(args.output / 'representatives-training-only.npy')
            or right['source_training_complete_sha256'] != sha(args.source / 'complete.json')
            or right['new_training_steps'] != 0 or not right['all_source_trunk_weights_exact'] or right['trunk_duplicated']):
        raise ValueError('Frozen source weights, features or PTQ pairing differs')
    result = {'scope': 'Export-only lexical product comparison on reused original validation; no new fitting or physical/final acceptance',
        'script_sha256': sha(Path(__file__)), 'plan_sha256': right['plan_sha256'],
        'same_trained_weights_features_and_ptq': True, 'new_training_steps': 0,
        'single_shared_trunk': True, **sides, 'speaker_playback': False, 'audio_speed_changed': False,
        'limits': ['Proxy-window metrics are not real-user recall or phonetic latency.',
            'Pointwise product bounds alone cannot guarantee fewer latched events; actual continuous event counters are evaluated.',
            'The auxiliary inference branch adds operators/parameters and requires its own P4 fit/numeric/resource validation.',
            'Output quantization is measured from this graph, not assumed to match a standalone sigmoid.'], 'accepted': False}
    with args.report.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: side[key] for key in ('confidence_and_confuser_qualified', 'confidence_only_diagnostic')}
                      for name, side in sides.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'compare'])
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.action == 'build' and args.plan is None or args.action == 'compare' and args.report is None:
        parser.error('Build requires --plan; comparison requires fresh --report')
    (build if args.action == 'build' else compare)(args)


if __name__ == '__main__':
    main()
