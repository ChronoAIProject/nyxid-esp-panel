"""Bounded eager/graph TF control for the first-batch derivative comparison.

The independent NumPy/encoder-backprop comparison retained two supervised
entries outside the original1e-5element tolerance. Reproduce full float32 TF
autodiff in eager and graph modes, without changing that result or tolerance.
This control distinguishes execution/derivative implementations descriptively;
it is not another training or an isolated kernel-cause experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "initialization", "mac-replay", "received", "v46-model", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output required"
    replay_bytes = (args.mac_replay / "report.json").read_bytes()
    replay = json.loads(replay_bytes)
    assert replay["original_probe_sha256"] == sha((args.probe / "gradient-probe.json").read_bytes())
    original = json.loads((args.probe / "gradient-probe.json").read_bytes())
    assert sha((args.probe / "probe-batches.npz").read_bytes()) == original["batches_sha256"]
    assert sha((args.initialization / "derived-probe-initialization.weights.h5").read_bytes()) == replay["derived_h5_sha256"]
    assert subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"]).decode().strip() == replay["upstream_revision"]
    for path, digest in replay["upstream_source_sha256"].items():
        assert sha((args.upstream / path).read_bytes()) == digest
    with np.load(args.probe / "probe-batches.npz", allow_pickle=False) as data:
        inputs = [data[k][0] for k in ("main_x", "main_y", "main_weight", "pair_x", "pair_y", "pair_weight")]
    references = {}
    for device in ("cpu", "cuda"):
        report = json.loads((args.received / device / "report.json").read_bytes())
        assert report["snapshot_sha256"] == replay["derived_h5_sha256"]
        assert report["batches_sha256"] == original["batches_sha256"]
        assert sha((args.received / device / "arrays.npz").read_bytes()) == report["arrays_sha256"]
        with np.load(args.received / device / "arrays.npz", allow_pickle=False) as data:
            references[device] = {k: data[k] for k in data.files}
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(args.upstream.resolve()))
    from microwakeword import mixednet
    training_bytes = (args.v46_model / "training.json").read_bytes()
    assert sha(training_bytes) == "bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858"
    options = argparse.ArgumentParser()
    mixednet.model_parameters(options)
    model = mixednet.model(options.parse_args(json.loads(training_bytes)["architecture_args"]), (150, 40), None)
    model.load_weights(args.initialization / "derived-probe-initialization.weights.h5")
    assert [dict(path=v.path, shape=list(v.shape), dtype=v.numpy().dtype.name,
                 value_sha256=sha(v.numpy().tobytes())) for v in model.variables] == json.loads(
                     (args.initialization / "variable-inventory.json").read_bytes())
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    before = [v.numpy().copy() for v in model.variables]
    head = model.layers[-1]
    variables = model.trainable_variables
    extractor = tf.keras.Model(model.input, head.input)

    def forward_backward(x, y, w, sx, sy, sw):
        with tf.GradientTape(persistent=True) as tape:
            em, ep = extractor(x, training=True), extractor(sx, training=True)
            main = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=y, logits=em @ head.kernel + head.bias)[:, 0] * w)
            pair = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=sy, logits=ep @ head.kernel + head.bias)[:, 0] * sw)
            supervised = main + pair
            values = tf.reshape(ep, (2, 4, 5, 300))
            unit = values / tf.sqrt(tf.maximum(tf.reduce_sum(values * values, axis=-1, keepdims=True),
                                               tf.constant(1e-16, dtype=values.dtype)))
            d = tf.reduce_mean(1 - tf.reduce_sum(unit[:, :, :1] * unit[:, :, 1:], axis=-1))
        gs, gd = tape.gradient(supervised, variables), tape.gradient(d, variables)
        assert all(v is not None for v in gs)
        gd = [tf.zeros_like(v) if g is None else g for v, g in zip(variables, gd)]
        return tf.stack([main, pair, d]), tf.concat([tf.reshape(g, [-1]) for g in gs], 0), tf.concat(
            [tf.reshape(g, [-1]) for g in gd], 0)

    tensors = [tf.convert_to_tensor(v) for v in inputs]
    compiled = tf.function(forward_backward)
    observed = {}
    for mode, function in (("eager", forward_backward), ("graph", compiled)):
        rows = [[v.numpy() for v in function(*tensors)] for _ in range(2)]
        for a, b in zip(*rows):
            np.testing.assert_array_equal(a, b)
        observed[mode] = dict(zip(("metrics", "supervised", "consistency"), rows[0]))
    for before_value, variable in zip(before, model.variables):
        np.testing.assert_array_equal(before_value, variable.numpy())
    args.output.mkdir(parents=True)
    packed = {mode + "_" + key: value for mode, values in observed.items() for key, value in values.items()}
    np.savez_compressed(args.output / "arrays.npz", **packed)
    comparisons = {}
    for mode, values in observed.items():
        comparisons[mode] = {}
        for device, expected in references.items():
            comparisons[mode][device] = {}
            for key, value in values.items():
                delta = np.abs(value.astype(np.float64) - expected[key].astype(np.float64))
                bound = 1e-5 + 1e-5 * np.abs(expected[key].astype(np.float64))
                comparisons[mode][device][key] = dict(passed=bool(np.all(delta <= bound)),
                    mismatched_elements=int(np.count_nonzero(delta > bound)),
                    maximum_absolute_error=float(delta.max()), maximum_fraction_of_fixed_bound=float((delta / bound).max()))
    result = dict(scope=__doc__, mac_reference_report_sha256=sha(replay_bytes),
                  source_visible_tf_graph=True, exact_repeat_both_modes=True, model_bn_values_unchanged=True,
                  optimizer_updates=0, atol=1e-5, rtol=1e-5, comparisons=comparisons,
                  metrics={k: v["metrics"].tolist() for k, v in observed.items()},
                  arrays_sha256=sha((args.output / "arrays.npz").read_bytes()),
                  script_sha256=sha(Path(__file__).read_bytes()), tensorflow_version=tf.__version__,
                  limits="First-batch controlled eager/graph full-TF execution, not unique operation causality, "
                         "a replacement for independent NumPy comparison or passage of old GPU failures. "
                         "Does not change fixed coefficient, training runs, event gates or deployment.")
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
