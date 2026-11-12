"""Measure erased training draws at derived initialization and rejected v46 weights.

This is a partial-loss diagnostic, not a new trained candidate or matched run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'complete-pack', 'initialization', 'v46-model', 'upstream', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = json.loads(args.preflight.read_text())
    prefix = 'research/wakeword_p4/training/'
    source = subprocess.check_output([
        'git', 'show', previous['source_commit'] + ':' + prefix + 'complete_coverage.py'])
    assert hashlib.sha256(source).hexdigest() == previous['source_sha256'][prefix + 'complete_coverage.py']
    derived = args.initialization / 'derived-probe-initialization.weights.h5'
    trained = args.v46_model / 'best.weights.h5'
    assert sha(derived) == '21f53feec03bdd73ae51cbc92091bf723d9de5fba07e3aa337c818ed722fe350'
    assert sha(trained) == '16b01e2f17473121c635801e324f5cf550c2ea0b72f8a054c04d845a0f596027'
    assert sha(args.v46_model / 'training.json') == 'bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858'
    upstream = args.upstream.resolve()
    revision = '4665173cd35f1cff9a61e06fc427f124766c488e'
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == revision
    for path in subprocess.check_output(['git', '-C', str(upstream), 'ls-files', 'microwakeword'], text=True).splitlines():
        if path.endswith('.py'):
            assert (upstream / path).read_bytes() == subprocess.check_output([
                'git', '-C', str(upstream), 'show', revision + ':' + path])
    with tempfile.TemporaryDirectory(prefix='nyx-recovery-model-') as temporary:
        filename = Path(temporary) / 'complete_coverage.py'
        filename.write_bytes(source)
        namespace = {'__file__': str(filename)}
        exec(compile(source, str(filename), 'exec'), namespace)
        sampler = namespace['CompleteCoverage'](
            args.complete_pack, negative_timeline=True, positive_timeline=True,
            negative_container_mixture=True)
    assert sampler.report['input_sha256'] == previous['complete_pack_sha256']
    windows, labels, engines = [], [], []
    engine_names = ['kokoro', 'macos-say', 'openai-tts', 'piper', 'qwen']
    for _ in range(50):
        ids = []
        draw = sampler.draw_indices

        def capture():
            chosen = draw()
            ids.extend(chosen)
            return chosen

        sampler.draw_indices = capture
        try:
            x = sampler.replace(np.zeros((80, 150, 40), np.float32))[np.r_[0:8, 32:44]]
        finally:
            sampler.draw_indices = draw
        chosen = np.asarray(ids)[np.r_[0:8, 16:28]]
        y = sampler.y[chosen]
        np.testing.assert_array_equal(y, [1] * 8 + [0] * 12)
        windows.append(x)
        labels.append(y)
        engines.append([engine_names.index(sampler.sources[sampler.records[i]['source_index']]['aliases'][0]['engine'])
                        for i in chosen])
    windows = np.stack(windows)
    labels = np.stack(labels).astype(np.float32)
    engines = np.asarray(engines, np.int32)
    assert hashlib.sha256(windows.tobytes()).hexdigest() == previous['recovered_exact_feature_sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet
    training = json.loads((args.v46_model / 'training.json').read_text())
    options = argparse.ArgumentParser()
    mixednet.model_parameters(options)
    model = mixednet.model(options.parse_args(training['architecture_args']), (150, 40), None)
    assert model.count_params() == 22801
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    head = model.layers[-1]
    extractor = tf.keras.Model(model.input, head.input)
    variables = model.trainable_variables

    @tf.function
    def observe(x, y):
        with tf.GradientTape() as tape:
            embedding = extractor(x, training=True)
            logits = (tf.matmul(embedding, head.kernel) + head.bias)[:, 0]
            weights = tf.where(y > 0, 1., 20.)
            # These20 samples would contribute sum(weighted BCE)/128 to the
            # ordinary main loss. The remaining108 and paired40 are excluded.
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) * weights) / 128.
        gradients = tape.gradient(loss, variables)
        assert all(g is not None for g in gradients)
        for gradient in gradients:
            tf.debugging.assert_all_finite(gradient, 'Recovered-window gradient is nonfinite')
        tf.debugging.assert_all_finite(logits, 'Recovered-window logit is nonfinite')
        return logits, loss, tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)

    args.output.mkdir(parents=True)
    details, reports = {}, {}
    for name, weights_path in [('derived_initialization', derived), ('rejected_v46', trained)]:
        model.load_weights(weights_path)
        before = [v.numpy().copy() for v in model.variables]
        logits, losses, gradients = [], [], []
        for i in range(50):
            z, loss, gradient = observe(windows[i], labels[i])
            logits.append(z.numpy())
            losses.append(float(loss))
            gradients.append(gradient.numpy())
            if (i + 1) % 10 == 0:
                print(json.dumps(dict(model=name, batches=i + 1, optimizer_updates=0)), flush=True)
        for variable, value in zip(model.variables, before, strict=True):
            np.testing.assert_array_equal(variable.numpy(), value)
        # Repeat the first batch to detect mutation/nondeterministic state.
        z, loss, gradient = observe(windows[0], labels[0])
        np.testing.assert_array_equal(z.numpy(), logits[0])
        np.testing.assert_array_equal(gradient.numpy(), gradients[0])
        logits = np.stack(logits).astype(np.float64)
        gradients = np.stack(gradients)
        weights = np.where(labels > 0, 1., 20.)
        bce = np.logaddexp(0., logits) - labels * logits
        probability = np.exp(-np.logaddexp(0., -logits))
        derivative = np.abs((probability - labels) * weights / 128.)
        by_engine = {}
        for e, engine in enumerate(engine_names):
            by_engine[engine] = {}
            for label, task in [(1, 'positive'), (0, 'confusing_negative')]:
                mask = (engines == e) & (labels == label)
                assert mask.any()
                by_engine[engine][task] = dict(
                    windows=int(mask.sum()), mean_unweighted_bce=float(bce[mask].mean()),
                    mean_probability=float(probability[mask].mean()),
                    wrong_logit_sign_windows=int(np.sum((logits[mask] >= 0) != bool(label))),
                    absolute_logit_derivative_mass=float(derivative[mask].sum()))
        reports[name] = dict(weights_sha256=sha(weights_path), by_engine=by_engine,
            mean_partial_main128_bce=float(np.mean(losses)),
            gradient_norm_min=float(np.linalg.norm(gradients.astype(np.float64), axis=1).min()),
            gradient_norm_max=float(np.linalg.norm(gradients.astype(np.float64), axis=1).max()),
            zero_gradient_batches=int(np.sum(np.all(gradients == 0, axis=1))),
            all_gradients_connected_finite=True, repeated_first_batch_exact=True,
            all_variables_and_bn_unchanged=True)
        details[name + '_logits'] = logits
        details[name + '_gradients'] = gradients
    np.savez_compressed(args.output / 'details.npz', labels=labels, engines=engines, **details)
    report = dict(scope=__doc__, script_sha256=sha(Path(__file__)),
        preflight_sha256=sha(args.preflight), feature_sha256=previous['recovered_exact_feature_sha256'],
        windows=1000, optimizer_updates=0, models=reports,
        details_sha256=sha(args.output / 'details.npz'), tensorflow=tf.__version__, numpy=np.__version__,
        upstream_revision=revision,
        limits='Training-source-only partial BCE over20 recovered windows per batch, divided by128. '
               'No actual main128/paired40 replacement, training update, new candidate, independent '
               'quality evaluation or current-GPU-result claim. Derived initialization is not a '
               'recovered original probe instance. V46 is already rejected; logit sign is a fixed '
               'training diagnostic, not a detector threshold. No coefficient/threshold tuning, '
               'new/final data, playback, recording, board or production operation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
