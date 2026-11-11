"""Verify shared sequence/window/stream scores and gradients on random inputs.

This accesses no speech, pretrained checkpoint, device or recognition set.
It proves a training graph mapping, not wake quality or hardware performance.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

from sequence_scorer import build_sequence_scorer

ARCH = ['--residual_connection', '0,0,0,0', '--stride', '3',
        '--pointwise_filters', '60,60,60,60', '--repeat_in_block', '1,1,1,1',
        '--mixconv_kernel_sizes', '[5], [9], [13], [21]',
        '--first_conv_filters', '30', '--first_conv_kernel_size', '5']
UPSTREAM = '4665173cd35f1cff9a61e06fc427f124766c488e'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, text=True).strip()
    assert revision == UPSTREAM
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(97727)
    sys.path.insert(0, str(a.upstream.resolve()))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    parser = argparse.ArgumentParser()
    mixednet.model_parameters(parser)
    flags = parser.parse_args(ARCH)
    frames, stride = 150, flags.stride
    effective = (frames-flags.first_conv_kernel_size)//stride*stride+flags.first_conv_kernel_size
    phase = (-effective) % stride
    assert (effective, phase) == (149, 1)
    window = mixednet.model(flags, (frames, 40), None)
    bn_rng = np.random.default_rng(97728)
    for layer in window.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            shape = layer.gamma.shape
            layer.set_weights([bn_rng.uniform(.4, 1.8, shape).astype(np.float32),
                bn_rng.uniform(-.3, .3, shape).astype(np.float32),
                bn_rng.uniform(-.2, .2, shape).astype(np.float32),
                bn_rng.uniform(.5, 2., shape).astype(np.float32)])
            layer.trainable = False
    # Keep the random sigmoid away from saturation for informative gradient checks.
    dense = window.layers[-1]
    dense.kernel.assign(tf.random.stateless_normal(dense.kernel.shape, [97727, 1], stddev=.05))
    dense.bias.assign(tf.zeros_like(dense.bias))
    sequence = build_sequence_scorer(window)
    assert sequence.count_params() == window.count_params() == 22801
    frozen_initial = [v.numpy().copy() for v in window.non_trainable_variables]

    @tf.function(input_signature=[tf.TensorSpec([None, None, 40], tf.float32)])
    def compiled_five_gradient(data):
        with tf.GradientTape() as tape:
            scores = sequence(data, training=True)
            objective = tf.reduce_mean(scores[:, 17:22])
        return scores, tape.gradient(objective, window.trainable_variables)
    # Upstream conversion changes Stream.mode on its source layers. Keep the
    # window/sequence training graph separate from the export graph.
    export_window = mixednet.model(flags, (frames, 40), None)
    export_window.set_weights(window.get_weights())
    streaming = utils.to_streaming_inference(export_window, {'stride': stride, 'spectrogram_length': frames},
                                             modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), np.float32))
    states = [v for v in streaming.variables if 'states' in v.name]
    assert len(states) == 6
    details = []
    rng = np.random.default_rng(97727)
    for count in (420, 423):
        x = rng.uniform(0, 26, (1, count, 40)).astype(np.float32)
        unshifted = sequence(x, training=False).numpy()[0]
        starts = np.arange(0, count-frames+1, stride)
        windows = np.stack([x[0, start:start+frames] for start in starts])
        w = window(windows, training=False).numpy()[:, 0]
        np.testing.assert_allclose(unshifted[:len(w)], w, rtol=1e-5, atol=2e-6)
        for state in states:
            state.assign(tf.zeros_like(state))
        streamed = np.asarray([streaming(x[:, i:i+stride], training=False).numpy()[0, 0]
                               for i in range(0, count, stride)])
        aligned = sequence(x[:, phase:], training=False).numpy()[0]
        first = (effective+phase)//stride-1
        mature = streamed[first:]
        np.testing.assert_allclose(aligned, mature, rtol=1e-5, atol=2e-6)
        # Compare the five-score training objective for every original variable.
        at = 17
        with tf.GradientTape() as tape:
            value = tf.reduce_mean(sequence(x, training=False)[0, at:at+5])
        sg = tape.gradient(value, window.trainable_variables)
        compiled_scores, compiled_gradients = compiled_five_gradient(tf.convert_to_tensor(x))
        np.testing.assert_allclose(compiled_scores.numpy()[0], unshifted, rtol=1e-5, atol=2e-6)
        compiled_deltas = []
        for s, compiled in zip(sg, compiled_gradients, strict=True):
            np.testing.assert_allclose(compiled.numpy(), s.numpy(), rtol=1e-4, atol=3e-6)
            compiled_deltas.append(float(np.max(np.abs(compiled.numpy()-s.numpy()))))
        for variable, initial in zip(window.non_trainable_variables, frozen_initial, strict=True):
            np.testing.assert_array_equal(variable.numpy(), initial)
        five_windows = np.stack([x[0, i*stride:i*stride+frames] for i in range(at, at+5)])
        with tf.GradientTape() as tape:
            value = tf.reduce_mean(window(five_windows, training=False))
        wg = tape.gradient(value, window.trainable_variables)
        gradient_deltas = []
        for variable, s, wgrad in zip(window.trainable_variables, sg, wg, strict=True):
            assert s is not None and wgrad is not None, variable.name
            assert np.isfinite(s.numpy()).all() and np.isfinite(wgrad.numpy()).all()
            np.testing.assert_allclose(s.numpy(), wgrad.numpy(), rtol=1e-4, atol=3e-6)
            gradient_deltas.append(float(np.max(np.abs(s.numpy()-wgrad.numpy()))))
        assert sum(float(np.sum(np.abs(g.numpy()))) for g in sg) > 1e-5
        details.append({'random_input_frames': count, 'input_sha256': sha(x.tobytes()),
            'compared_window_scores': len(w), 'compared_mature_stream_scores': len(mature),
            'window_max_abs_delta': float(np.max(np.abs(unshifted[:len(w)]-w))),
            'stream_max_abs_delta': float(np.max(np.abs(aligned-mature))),
            'max_gradient_abs_delta': max(gradient_deltas), 'gradient_variables': len(sg),
            'compiled_training_max_gradient_abs_delta': max(compiled_deltas),
            'non_saturated_score_min': float(unshifted.min()), 'score_max': float(unshifted.max())})
    result = {'scope': __doc__, 'upstream_revision': revision, 'architecture_args': ARCH,
        'seed': 97727, 'parameters': window.count_params(), 'window_frames': frames,
        'effective_input_frames': effective, 'stream_phase_frames': phase,
        'first_mature_stream_output_zero_based': first,
        'same_original_trainable_variables': True, 'optimizer_steps': 0,
        'nonidentity_frozen_batchnorm_seed': 97728,
        'gradient_objective': 'Mean of five consecutive scores versus five separate overlapping windows',
        'compiled_dynamic_length_training_true_frozen_bn_unchanged': True,
        'tensorflow': version('tensorflow'), 'numpy': version('numpy'),
        'python': sys.version, 'platform': platform.platform(), 'details': details,
        'sources_sha256': {name: sha(Path(__file__).with_name(name).read_bytes())
                           for name in ('sequence_scorer.py', 'probe_sequence_scorer.py')},
        'limits': 'Random initialized teacher60 only. No source audio or checkpoint, training, INT8 export, MCU, recognition, timing or resource acceptance.'}
    with a.output.open('x') as out:
        out.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
