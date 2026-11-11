"""Execute the pinned GPU combined-update method on informative random CPU data.

No speech/checkpoint/feature pack is opened. The data-authentication constructor
and sampler are deliberately outside this probe; the actual compiled update
method is loaded unchanged from the pinned research commit.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf

from probe_sequence_scorer import ARCH, UPSTREAM
from sequence_loss import build_sequence_logits, mean_probability_bce

GPU_REVISION = 'e491229d6d608a35ec3a1393eeefdfc600e5a71c'
GPU_SOURCE = 'research/wakeword_p4/training/five_score_training.py'
GPU_SOURCE_SHA = '81445b8774c37aaeb43fdb58e52ba6226ebf481332f704d7f597af1c9b3907b2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RecordingAdam(tf.keras.optimizers.Adam):
    """Record incoming gradients; delegate the actual update to ordinary Adam."""
    def build(self, variables):
        if self.built:
            return
        super().build(variables)
        self.recorded = [self.add_variable_from_reference(v, name='probe_gradient') for v in variables]

    def apply_gradients(self, grads_and_vars):
        pairs = list(grads_and_vars)
        self.target_ids = [id(v) for _, v in pairs]
        for slot, (gradient, _) in zip(self.recorded, pairs, strict=True):
            slot.assign(gradient)
        return super().apply_gradients(pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(['git', 'show', GPU_REVISION+':'+GPU_SOURCE], cwd=repo)
    assert sha(source) == GPU_SOURCE_SHA
    helper = subprocess.check_output(['git', 'show', GPU_REVISION+':research/wakeword_p4/training/sequence_loss.py'], cwd=repo)
    assert helper == Path(__file__).with_name('sequence_loss.py').read_bytes()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip()
    assert revision == UPSTREAM
    args.output.mkdir(parents=True, exist_ok=False)
    source_path = args.output/'five_score_training.py'
    source_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location('pinned_gpu_five_score', source_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(97731)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    parser = argparse.ArgumentParser()
    mixednet.model_parameters(parser)
    flags = parser.parse_args(ARCH)
    model = mixednet.model(flags, (150, 40), None)
    rng = np.random.default_rng(97732)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            shape = layer.gamma.shape
            layer.set_weights([rng.uniform(.4, 1.8, shape).astype(np.float32),
                rng.uniform(-.3, .3, shape).astype(np.float32),
                rng.uniform(-.2, .2, shape).astype(np.float32),
                rng.uniform(.5, 2., shape).astype(np.float32)])
            layer.trainable = False
    model.layers[-1].kernel.assign(tf.random.stateless_normal(model.layers[-1].kernel.shape, [97731, 1], stddev=.05))
    model.layers[-1].bias.assign(tf.zeros_like(model.layers[-1].bias))
    reference = mixednet.model(flags, (150, 40), None)
    reference.set_weights(model.get_weights())
    for layer in reference.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    assert model.count_params() == reference.count_params() == 22801
    initial = [v.numpy().copy() for v in model.variables]
    model.save_weights(args.output/'initial.weights.h5')
    frozen = [v.numpy().copy() for v in model.non_trainable_variables]
    optimizer = RecordingAdam(1e-3, global_clipnorm=5)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    optimizer.build(model.trainable_variables)
    worker = module.FiveScoreTraining.__new__(module.FiveScoreTraining)
    worker.model = model
    worker.logits = build_sequence_logits(model)
    x = rng.uniform(0, 26, (128, 150, 40)).astype(np.float32)
    y = np.asarray([1.]*32+[0.]*96, dtype=np.float32)[:, None]
    weights = np.where(y[:, 0] > 0, 1., 20.).astype(np.float32)
    sx = rng.uniform(0, 26, (16, 162, 40)).astype(np.float32)
    sy = np.asarray([1.]*8+[0.]*8, dtype=np.float32)
    sw = np.where(sy > 0, 1., 20.).astype(np.float32)
    trunk = tf.keras.Model(reference.input, reference.layers[-2].output)
    dense = reference.layers[-1]

    @tf.function
    def direct_gradient(x, y, weights, sx, sy, sw):
        with tf.GradientTape(persistent=True) as tape:
            main = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, reference(x, training=True))*weights)
            logits = tf.stack([tf.reshape(tf.matmul(trunk(sx[:, at:at+150], training=True), dense.kernel)+dense.bias, [-1])
                               for at in range(0, 13, 3)], axis=1)
            extra = tf.reduce_mean(mean_probability_bce(sy, logits)*sw)
            loss = main+extra
        return main, extra, loss, logits, tape.gradient(loss, reference.trainable_variables), tape.gradient(extra, reference.trainable_variables)

    main_loss, extra_loss, direct_loss, logits, gradients, extra_gradients = direct_gradient(x, y, weights, sx, sy, sw)
    loss = worker.step(x, y, weights, sx, sy, sw)
    assert optimizer.target_ids == [id(v) for v in model.trainable_variables]
    assert int(optimizer.iterations) == 1
    actual_gradients = [g.numpy().copy() for g in optimizer.recorded]
    gradient_errors = []
    for expected, actual in zip(gradients, actual_gradients, strict=True):
        assert np.isfinite(expected.numpy()).all() and np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        gradient_errors.append(float(np.max(np.abs(actual-expected.numpy()))))
    loss_error = abs(float(loss)-float(direct_loss))
    assert loss_error < 1e-4
    extra_l1 = sum(float(tf.reduce_sum(tf.abs(g))) for g in extra_gradients)
    assert extra_l1 > 1e-3
    reference_optimizer = tf.keras.optimizers.Adam(1e-3, global_clipnorm=5)
    reference_optimizer.apply_gradients(zip([tf.constant(g) for g in actual_gradients], reference.trainable_variables))
    update_errors = [float(np.max(np.abs(a.numpy()-b.numpy())))
                     for a, b in zip(model.variables, reference.variables, strict=True)]
    assert max(update_errors) == 0
    assert all(np.array_equal(v.numpy(), original) for v, original in zip(model.non_trainable_variables, frozen, strict=True))
    changes = [float(np.max(np.abs(v.numpy()-original))) for v, original in zip(model.variables, initial, strict=True)]
    assert max(changes) > 0
    arrays = {'x': x, 'y': y, 'weights': weights, 'sx': sx, 'sy': sy, 'sw': sw, 'direct_logits': logits.numpy()}
    for i, (actual, expected) in enumerate(zip(actual_gradients, gradients, strict=True)):
        arrays['actual_gradient_'+str(i)] = actual
        arrays['reference_gradient_'+str(i)] = expected.numpy()
    np.savez_compressed(args.output/'arrays.npz', **arrays)
    result = {'scope': __doc__, 'gpu_source_commit': GPU_REVISION, 'gpu_source_sha256': sha(source),
        'probe_source_sha256': sha(Path(__file__).read_bytes()), 'upstream_revision': revision,
        'sequence_loss_sha256': sha(helper), 'tensorflow': tf.__version__,
        'initial_weights_sha256': sha((args.output/'initial.weights.h5').read_bytes()),
        'arrays_sha256': sha((args.output/'arrays.npz').read_bytes()),
        'main_positive_negative': [32, 96], 'sequence_positive_negative': [8, 8],
        'negative_weight': 20, 'sequence_loss_weight': 1,
        'main_loss': float(main_loss), 'extra_loss': float(extra_loss),
        'combined_loss': float(loss), 'direct_loss': float(direct_loss), 'loss_max_delta': loss_error,
        'extra_gradient_l1': extra_l1, 'gradient_variables': len(gradients),
        'gradient_max_deltas': gradient_errors, 'parameter_update_max_deltas': update_errors,
        'optimizer_iteration_delta': int(optimizer.iterations), 'frozen_bn_unchanged': True,
        'all_original_variables_used': True, 'maximum_parameter_change': max(changes),
        'candidate_promoted': False,
        'limits': ['Informative random CPU teacher60 graph, not actual GPU initialization/checkpoint/data or a recognition candidate.',
            'Constructor/pack authentication and sampler are bypassed; this exercises only the pinned unchanged compiled step.',
            'Recording Adam only adds gradient-capture slots; its update is checked against ordinary Adam with the same incoming gradients.',
            'No GPU/P4, recognition, FAR, latency, audio, device state or final/new-source evidence.']}
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
