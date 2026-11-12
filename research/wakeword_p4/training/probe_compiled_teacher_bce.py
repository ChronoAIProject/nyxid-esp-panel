"""Check the actual compiled teacher BCE recipe on artificial saturated heads.

No speech, checkpoint, score dataset, optimizer update or device is used.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
import keras


UPSTREAM = '4665173cd35f1cff9a61e06fc427f124766c488e'
BASE = 'research/wakeword_p4/training/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--gpu-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    revision = subprocess.check_output(
        ['git', 'rev-parse', args.gpu_revision+'^{commit}'], text=True).strip()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip()
    assert actual == UPSTREAM
    mixednet_bytes = (args.upstream/'microwakeword/mixednet.py').read_bytes()
    assert mixednet_bytes == subprocess.check_output(
        ['git', 'show', UPSTREAM+':microwakeword/mixednet.py'], cwd=args.upstream)

    def read(name):
        return subprocess.check_output(['git', 'show', revision+':'+BASE+name])

    source, policy_bytes = read('train_stream.py'), read('batchnorm_policy.py')
    training_bytes = subprocess.check_output(['git', 'show', revision+
        ':research/wakeword_p4/results/4090-v41-training.json'])
    training = json.loads(training_bytes)
    assert training['architecture'] == 'teacher60' and training['context_frames'] == 150
    policy = {}
    exec(compile(policy_bytes, 'gpu-batchnorm-policy.py', 'exec'), policy)
    tree = ast.parse(source)
    body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    # Execute the original two recipe statements, not a hand-rewritten BCE.
    loss = next(node for node in body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'loss' for t in node.targets))
    compile_call = next(node for node in body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'model'
        and node.value.func.attr == 'compile')
    recipe = ast.Module(body=[loss, compile_call], type_ignores=[])
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    flags_parser = argparse.ArgumentParser()
    mixednet.model_parameters(flags_parser)
    model = mixednet.model(flags_parser.parse_args(training['architecture_args']), (150, 40), None)
    policy['apply_policy'](model, 'frozen')
    namespace = dict(tf=tf, model=model, args=SimpleNamespace(label_smoothing=0.))
    exec(compile(recipe, 'original-teacher-compile-recipe.py', 'exec'), namespace)
    assert model.loss == 'binary_crossentropy' and model.optimizer.global_clipnorm == 5
    head = model.layers[-1]
    assert isinstance(head, tf.keras.layers.Dense)
    assert tf.keras.activations.serialize(head.activation) == 'sigmoid'
    original = [v.numpy().copy() for v in model.variables]
    features = tf.keras.Model(model.input, head.input)
    x = tf.random.stateless_uniform((2, 150, 40), [726, 44], minval=0., maxval=26.)
    y, weights = tf.constant([[0.], [1.]]), tf.constant([20., 1.])

    def compare():
        with tf.GradientTape() as tape:
            prediction = model(x, training=True)
            actual_loss = model.compute_loss(x=x, y=y, y_pred=prediction, sample_weight=weights)
        gradients = tape.gradient(actual_loss, model.trainable_variables)
        with tf.GradientTape() as tape:
            logits = tf.matmul(features(x, training=True), head.kernel)+head.bias
            reference = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)[:, 0]*weights)
        reference_gradients = tape.gradient(reference, model.trainable_variables)
        return prediction, actual_loss, reference, gradients, reference_gradients

    compiled = tf.function(compare)
    details = []
    for case in ['original_random_head', 0., 1.25, -3.5, 20., 80., 1000., -20., -80., -1000.]:
        if case != 'original_random_head':
            head.kernel.assign(tf.zeros_like(head.kernel))
            head.bias.assign([case])
        before = [v.numpy().copy() for v in model.variables]
        row = dict(case=case)
        for mode, call in [('eager', compare), ('compiled', compiled)]:
            prediction, actual_loss, reference, gradients, expected = call()
            assert len(gradients) == len(expected) == len(model.trainable_variables)
            maximum = 0.
            for actual_gradient, expected_gradient in zip(gradients, expected, strict=True):
                assert actual_gradient is not None and expected_gradient is not None
                a, b = actual_gradient.numpy(), expected_gradient.numpy()
                assert np.isfinite(a).all() and np.isfinite(b).all()
                np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
                maximum = max(maximum, float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(actual_loss, reference, atol=1e-5, rtol=1e-5)
            bias_index = next(i for i, v in enumerate(model.trainable_variables) if v is head.bias)
            if isinstance(case, float) and abs(case) >= 20:
                assert abs(float(gradients[bias_index].numpy()[0])) > .49
            row[mode] = dict(prediction=prediction.numpy().tolist(), loss=float(actual_loss),
                logits_reference_loss=float(reference), gradient_variables=len(gradients),
                maximum_gradient_error=maximum, bias_gradient=gradients[bias_index].numpy().tolist())
            for variable, prior in zip(model.variables, before, strict=True):
                np.testing.assert_array_equal(variable.numpy(), prior)
            assert int(model.optimizer.iterations.numpy()) == 0
        details.append(row)
    for variable, prior in zip(model.variables, original, strict=True):
        variable.assign(prior)
    for variable, prior in zip(model.variables, original, strict=True):
        np.testing.assert_array_equal(variable.numpy(), prior)
    result = dict(
        scope='Actual teacher60 compiled BCE recipe, random features/weights and forced saturated heads only.',
        accepted=False, gpu_revision=revision, upstream_revision=actual,
        tensorflow=tf.__version__, keras=keras.__version__, numpy=np.__version__,
        architecture_args=training['architecture_args'], context_frames=150,
        source_sha256=sha(Path(__file__).read_bytes()), training_source_sha256=sha(source),
        policy_source_sha256=sha(policy_bytes), architecture_report_sha256=sha(training_bytes),
        mixednet_source_sha256=sha(mixednet_bytes),
        compile_recipe_ast_sha256=sha(ast.dump(recipe, include_attributes=False).encode()),
        random_input_sha256=sha(x.numpy().tobytes()), seed=726,
        optimizer_updates=0, all_diagnostic_calls_preserve_model_variables=True,
        original_random_variables_restored=True, details=details,
        conclusion='The tested actual terminal Dense-sigmoid BCE path preserves correcting gradients even at rounded probabilities0/1; no main-loss change is justified by a probability-clipping assumption.',
        limits=[
            'Artificial cases, not trained v41/v42 weights or actual training-example gradient coverage.',
            'Validates this Mac TensorFlow/Keras environment; GPU Keras version/execution not independently verified.',
            'Pre-optimizer gradients; existing global_clipnorm5 is retained, no optimizer step performed.',
            'No model-quality, generalization, quantization, speech, reserved-set or P4 evidence.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(cases=len(details), modes=['eager', 'compiled'], optimizer_updates=0,
                         conclusion=result['conclusion']), indent=2))


if __name__ == '__main__':
    main()
