"""Recover a training-only word-CTC encoder into the original scalar graph.

Only the shared encoder is imported. The original disconnected binary
classifier and a fresh downstream optimizer retain their normal definitions.
Synthetic probe values verify graph/weight conversion, not wake quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from phrase_ctc_readout import sha, sha_array, weights_sha


def attach_head(binary, seed=919):
    import tensorflow as tf
    classifier = binary.get_layer('classifier')
    states = tf.keras.layers.Reshape((67, 60))(classifier.input)
    encoder = tf.keras.Model(binary.input, states)
    x = tf.keras.Input((67, 60))
    head = tf.keras.Model(x, tf.keras.layers.Dense(6,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x))
    return encoder, head, tf.keras.Model(encoder.input, head(encoder.output))


def restore_encoder(binary, checkpoint, evidence):
    """Load the nested pretraining graph; require every tensor-group identity."""
    classifier = binary.get_layer('classifier')
    classifier_before = weights_sha(classifier.weights)
    encoder, head, full = attach_head(binary)
    if (not evidence['classifier_unchanged']
            or weights_sha(encoder.weights) != evidence['encoder_weights_before_sha256']
            or classifier_before != evidence['classifier_weights_sha256']
            or sha(checkpoint) != evidence['checkpoint_sha256']):
        raise ValueError('Original encoder/classifier or checkpoint differs from pretraining')
    full.load_weights(checkpoint)
    if (weights_sha(encoder.weights) != evidence['encoder_weights_after_sha256']
            or weights_sha(head.weights) != evidence['head_weights_sha256']
            or weights_sha(classifier.weights) != classifier_before):
        raise ValueError('Imported encoder/head or preserved scalar classifier differs')
    return encoder, head, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('upstream', 'initial-weights', 'pretraining', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory')
    plan = json.loads((args.pretraining/'plan.json').read_bytes())
    evidence = json.loads((args.pretraining/'full-fit-words.json').read_bytes())
    checkpoint = args.pretraining/evidence['checkpoint_file']
    if (sha(args.initial_weights) != plan['initializer_sha256']
            or subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'],
                text=True).strip() != plan['upstream_commit']
            or subprocess.check_output(['git', '-C', str(args.upstream), 'diff', 'HEAD'])):
        raise ValueError('Original initializer or upstream differs')
    if (len(evidence['train_ids']) != plan['unique_crops']
            or evidence['train_ids'] != list(range(plan['unique_crops']))
            or evidence['test_ids'] != evidence['train_ids']):
        raise ValueError('Expected the complete original training-only word fit')
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(3)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from transfer import build_model
    binary, _ = build_model(335)
    binary.load_weights(args.initial_weights)
    binary_before = weights_sha(binary.weights)
    parameter_count = binary.count_params()
    encoder, head, full = restore_encoder(binary, checkpoint, evidence)
    probes = np.random.default_rng(1863).uniform(0, 26, (4, 335, 40)).astype(np.float32)
    probe_encoder = np.asarray(encoder(probes, training=False))
    probe_binary = np.asarray(binary(probes, training=False))
    probe_word = np.asarray(full(probes, training=False))
    assert binary.output_shape == (None, 1) and binary.count_params() == parameter_count == 25831
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output/'binary.weights.h5'
    binary.save_weights(target)
    recovered, _ = build_model(335)
    recovered.load_weights(target)
    assert weights_sha(recovered.weights) == weights_sha(binary.weights)
    assert np.array_equal(np.asarray(recovered(probes, training=False)), probe_binary)
    recovered_encoder = tf.keras.Model(recovered.input,
        tf.keras.layers.Reshape((67, 60))(recovered.get_layer('classifier').input))
    assert np.array_equal(np.asarray(recovered_encoder(probes, training=False)), probe_encoder)
    report = {
        'scope': __doc__, 'source_sha256': sha(Path(__file__)),
        'pretraining_plan_sha256': sha(args.pretraining/'plan.json'),
        'pretraining_report_sha256': sha(args.pretraining/'full-fit-words.json'),
        'pretraining_checkpoint_sha256': sha(checkpoint),
        'original_binary_file_sha256': sha(args.initial_weights),
        'binary_weights_before_sha256': binary_before,
        'binary_weights_after_sha256': weights_sha(binary.weights),
        'encoder_weights_before_sha256': evidence['encoder_weights_before_sha256'],
        'encoder_weights_after_sha256': weights_sha(encoder.weights),
        'original_classifier_sha256': weights_sha(binary.get_layer('classifier').weights),
        'word_head_sha256': weights_sha(head.weights),
        'output_file_sha256': sha(target), 'parameters': parameter_count,
        'scalar_classifier_preserved': True, 'roundtrip_tensors_and_probes_exact': True,
        'synthetic_probe_input_sha256': sha_array(probes),
        'synthetic_probe_encoder_sha256': sha_array(probe_encoder),
        'synthetic_probe_binary_sha256': sha_array(probe_binary),
        'synthetic_probe_word_logits_sha256': sha_array(probe_word),
        'limits': 'No fitting, real feature/audio scoring, threshold choice or wake-quality evidence. Only weights are retained; no pretraining optimizer or word head enters scalar inference. Parent training-corpus exposure and weak held-engine generalization remain. No audio device, board, firmware or capture-path change.',
    }
    (args.output/'conversion.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
