"""Training-only engine adversary and fixed paired lexical comparison.

The adversary learns engine classification normally; its shared-feature
gradient is reversed at the predeclared scale. Nothing is added at inference.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ENGINES = ['edge-tts', 'kokoro', 'macos-say', 'openai-tts', 'piper', 'qwen']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reverse_gradient(values, scale):
    import tensorflow as tf
    if scale not in (0., .05):
        raise ValueError('Expected fixed zero/.05 gradient reversal')

    @tf.custom_gradient
    def reverse(x):
        return tf.identity(x), lambda gradient: -tf.cast(scale, gradient.dtype) * gradient

    return reverse(values)


def engine_labels(draws, by_id):
    result = []
    for identity, source, cls in draws:
        row = by_id[identity]
        if row['source_sha256'] != source or bool(row['label']) != (cls == 0):
            raise ValueError('Auxiliary draw source or binary role differs')
        result.append(ENGINES.index(row['engine']))
    return np.asarray(result, np.int32)


def compare(args):
    from compare_lexical_auxiliary import read_arm
    control, _, left = read_arm(args.control)
    treatment, _, right = read_arm(args.treatment)
    difference = 'domain_adversarial_scale'
    if control['plan'][difference] != 0 or treatment['plan'][difference] != .05:
        raise ValueError('Expected frozen zero/.05 paired experiment')
    if ({k: v for k, v in control['plan'].items() if k != difference} !=
            {k: v for k, v in treatment['plan'].items() if k != difference}):
        raise ValueError('Training plans differ beyond the engine adversary')
    for key in ('main_batches_sha256', 'auxiliary_batches_sha256', 'auxiliary_source_batches_sha256',
                'engine_labels_batches_sha256', 'initial_weights_sha256', 'auxiliary_manifest_sha256',
                'feature_inputs', 'parameters', 'training_sources', 'training_frames', 'engine_class_order'):
        if control[key] != treatment[key]:
            raise ValueError('Matched training evidence differs: ' + key)
    if (left['representatives_sha256'] != right['representatives_sha256']
            or left['export']['ops'] != right['export']['ops']
            or left['export']['model_bytes'] != right['export']['model_bytes']
            or control['domain_parameters_training_only'] != 0
            or treatment['domain_parameters_training_only'] != 390):
        raise ValueError('Matched PTQ, exported graph or training-only parameters differ')
    reference = json.loads(args.reference_complete.read_text())
    if sha(args.reference_complete) != control['plan']['retained_reference_complete_sha256']:
        raise ValueError('Frozen reference completion artifact differs')
    for key in ('main_batches_sha256', 'auxiliary_batches_sha256'):
        if control[key] != reference[key]:
            raise ValueError('New control does not reproduce retained training batches')
    if control['checkpoints'][0]['model_sha256'] != reference['checkpoints'][0]['model_sha256']:
        raise ValueError('New control does not reproduce retained model bytes')
    for side, complete in [(left, control), (right, treatment)]:
        for key in ('auxiliary_source_batches_sha256', 'engine_labels_batches_sha256',
                    'domain_parameters_training_only', 'engine_class_order'):
            side[key] = complete[key]
    a, b = left['confidence_and_confuser_qualified'], right['confidence_and_confuser_qualified']
    result = {'scope': 'Matched training-only engine-adversary ablation on reused validation, not physical/final acceptance',
              'comparison_script_sha256': sha(Path(__file__)), 'control': left, 'treatment': right,
              'matched_main_auxiliary_source_class_engine_and_ptq_verified': True,
              'control_model_and_main_auxiliary_batches_exactly_reproduced': True,
              'reference_complete_sha256': sha(args.reference_complete),
              'joint_qualified_macro_difference': b['macro_recall'] - a['macro_recall'] if a and b else None,
              'speaker_playback': False, 'microphone_access': False, 'audio_speed_changed': False,
              'accepted': False,
              'limits': ['Same synthetic sources, original uncertain anchors and reused validation; no independent speaker or Hi ESP parity claim.',
                         'Engine invariance is a training hypothesis, not proof of acoustic-domain generalization or correct pronunciation.',
                         'Only the original binary graph is exported; same shape does not prove MCU numeric/resource or voice stability acceptance.',
                         'Keep the complete fixed threshold grid; no acoustic or reserved-final scoring of an inferior candidate.']}
    with args.report.open('x') as out:
        out.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'control': a, 'treatment': b,
                      'difference': result['joint_qualified_macro_difference']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control', 'treatment', 'reference-complete', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    compare(parser.parse_args())
