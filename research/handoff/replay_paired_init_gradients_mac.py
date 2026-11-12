"""Replay all50 original batches with a certified derived initialization on Mac.

The original historical instance H5 was not saved. Authenticate the delivered
derived state against the original encoder/BN initializer and original head
bytes, then run CPU encoder autodiff with independent NumPy loss derivatives.
Report cross-runtime gradient differences descriptively, without fitting a
tolerance/weight or reclassifying an original failed GPU smoke.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from probe_rendering_consistency_gradient_mac import objective_and_gradient


PROBE = "0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f"
INITIALIZER = "8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b"
HEAD = "a41cc84a5e15a42a13ee9fe02e38242fca4dd9afe67a19ba736587970b46fc54"
DERIVED = "21f53feec03bdd73ae51cbc92091bf723d9de5fba07e3aa337c818ed722fe350"
INVENTORY = "04c209d4bc528a39d3ca1f9dd91ef6c0d1c0e8e24c9210819f4d09d5f2c9f648"
UPSTREAM = "4665173cd35f1cff9a61e06fc427f124766c488e"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def variable_inventory(model):
    return [dict(path=v.path, shape=list(v.shape), dtype=v.numpy().dtype.name,
                 value_sha256=sha(v.numpy().tobytes())) for v in model.variables]


def comparisons(actual, reference):
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    delta = actual - reference
    na, nr = np.linalg.norm(actual), np.linalg.norm(reference)
    assert np.isfinite(actual).all() and nr > 0 and na > 0
    return dict(actual_norm=float(na), original_gpu_norm=float(nr),
                relative_l2_error=float(np.linalg.norm(delta) / nr),
                maximum_absolute_error=float(np.max(np.abs(delta))),
                cosine=float(np.sum(actual * reference) / (na * nr)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe", "initialization", "v46-model", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output directory required"
    probe_bytes = (args.probe / "gradient-probe.json").read_bytes()
    assert sha(probe_bytes) == PROBE
    original = json.loads(probe_bytes)
    for name, field in (("probe-gradients.npz", "gradients_sha256"),
                        ("probe-batches.npz", "batches_sha256"),
                        ("initialization.json", "initialization_sha256")):
        assert sha((args.probe / name).read_bytes()) == original[field]
    initial = args.initialization
    for name, digest in (("original-reconstructed.weights.h5", INITIALIZER),
                         ("derived-probe-initialization.weights.h5", DERIVED),
                         ("variable-inventory.json", INVENTORY)):
        assert sha((initial / name).read_bytes()) == digest
    training_bytes = (args.v46_model / "training.json").read_bytes()
    assert sha(training_bytes) == "bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858"
    training = json.loads(training_bytes)
    upstream = args.upstream.resolve()
    assert subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"]).decode().strip() == UPSTREAM
    source_hashes = {}
    for path in subprocess.check_output(["git", "-C", str(upstream), "ls-files", "microwakeword"]).decode().splitlines():
        if path.endswith(".py"):
            data = (upstream / path).read_bytes()
            assert data == subprocess.check_output(["git", "-C", str(upstream), "show", UPSTREAM + ":" + path])
            source_hashes[path] = sha(data)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.backend.clear_session()
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet
    options = argparse.ArgumentParser()
    mixednet.model_parameters(options)
    model = mixednet.model(options.parse_args(training["architecture_args"]), (150, 40), None)
    assert model.count_params() == 22801
    model.load_weights(initial / "original-reconstructed.weights.h5")
    head = model.layers[-1]
    assert isinstance(head, tf.keras.layers.Dense) and tuple(head.input.shape) == (None, 300)
    head_ids = {id(v) for v in head.variables}
    source_values = {v.path: v.numpy().copy() for v in model.variables if id(v) not in head_ids}
    model.load_weights(initial / "derived-probe-initialization.weights.h5")
    for variable in model.variables:
        if id(variable) not in head_ids:
            np.testing.assert_array_equal(variable.numpy(), source_values[variable.path])
    assert len(source_values) == 29
    assert sha(b"".join(v.tobytes() for v in head.get_weights())) == HEAD
    assert variable_inventory(model) == json.loads((initial / "variable-inventory.json").read_bytes())
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    before = variable_inventory(model)
    variables = model.trainable_variables
    assert [(v.path, list(v.shape)) for v in variables] == [
        (v["path"], v["shape"]) for v in original["variable_layout"]]
    encoder = [v for v in variables if id(v) not in head_ids]
    encoder_ids = {id(v) for v in encoder}
    extractor = tf.keras.Model(model.input, head.input)
    kernel, bias = [v.astype(np.float64) for v in head.get_weights()]
    with np.load(args.probe / "probe-batches.npz", allow_pickle=False) as data:
        batches = {k: data[k] for k in data.files}
    with np.load(args.probe / "probe-gradients.npz", allow_pickle=False) as data:
        references = {k: data[k] for k in data.files}
    output_gradients = {k: [] for k in ("main", "pair", "supervised", "consistency")}
    traces = []

    def binary(embedding, labels, weights):
        e = embedding.astype(np.float64)
        z = (e @ kernel + bias)[:, 0]
        y = labels[:, 0].astype(np.float64)
        w = weights.astype(np.float64)
        probability = np.exp(-np.logaddexp(0., -z))
        dz = (probability - y) * w / len(y)
        loss = np.mean(w * (np.logaddexp(0., z) - y * z))
        return float(loss), (dz[:, None] * kernel[:, 0]).astype(np.float32), [
            (e.T @ dz[:, None]).astype(np.float32), np.array([dz.sum()], np.float32)]

    def full_gradient(encoder_gradients, head_gradients):
        assert all(g is not None for g in encoder_gradients)
        by_id = {id(v): g.numpy() for v, g in zip(encoder, encoder_gradients)}
        by_id.update({id(v): g for v, g in zip(head.trainable_variables, head_gradients)})
        return np.concatenate([by_id[id(v)].reshape(-1) for v in variables]).astype(np.float32)

    for i in range(50):
        x, sx = [tf.convert_to_tensor(batches[k][i]) for k in ("main_x", "pair_x")]
        with tf.GradientTape(persistent=True) as tape:
            em = extractor(x, training=True)
            ep = extractor(sx, training=True)
        main, dem, gm_head = binary(em.numpy(), batches["main_y"][i], batches["main_weight"][i])
        pair, dep, gp_head = binary(ep.numpy(), batches["pair_y"][i], batches["pair_weight"][i])
        d, ded = objective_and_gradient(ep.numpy().reshape(2, 4, 5, 300))
        gm = full_gradient(tape.gradient(em, encoder, output_gradients=tf.convert_to_tensor(dem)), gm_head)
        gp = full_gradient(tape.gradient(ep, encoder, output_gradients=tf.convert_to_tensor(dep)), gp_head)
        gs = full_gradient(tape.gradient([em, ep], encoder, output_gradients=[
            tf.convert_to_tensor(dem), tf.convert_to_tensor(dep)]),
            [a + b for a, b in zip(gm_head, gp_head)])
        gd = full_gradient(tape.gradient(ep, encoder, output_gradients=tf.convert_to_tensor(
            ded.reshape(40, 300).astype(np.float32))), [np.zeros_like(v, dtype=np.float32) for v in (kernel, bias)])
        del tape
        row = dict(step=i + 1, main_bce=main, pair_bce=pair, consistency=d,
                   loss_difference_from_original_gpu={
                       k: float(value - original["traces"][i][k]) for k, value in (
                           ("main_bce", main), ("pair_bce", pair), ("consistency", d))},
                   gradient_comparisons={})
        for name, value in (("main", gm), ("pair", gp), ("supervised", gs), ("consistency", gd)):
            output_gradients[name].append(value)
            row["gradient_comparisons"][name] = comparisons(value, references[name][i])
        traces.append(row)
        if (i + 1) % 10 == 0:
            print(json.dumps(dict(phase="original_batch_cpu_autodiff", batches=i + 1)), flush=True)
    assert variable_inventory(model) == before, "Read-only gradient replay changed model/BN"
    mask = np.concatenate([np.repeat(id(v) in encoder_ids, int(np.prod(v.shape))) for v in variables])
    output_gradients = {k: np.stack(v) for k, v in output_gradients.items()}
    ns = np.linalg.norm(output_gradients["supervised"][:, mask].astype(np.float64), axis=1)
    nd = np.linalg.norm(output_gradients["consistency"][:, mask].astype(np.float64), axis=1)
    assert np.all(ns > 0) and np.all(nd > 0)
    ratio = float(np.median(ns) / np.median(nd))
    coefficient = min(1., 10. ** np.floor(np.log10(.1 * ratio)))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "mac-gradients.npz", **output_gradients)
    result = dict(scope=__doc__, original_probe_sha256=PROBE, derived_h5_sha256=DERIVED,
                  original_initializer_sha256=INITIALIZER, original_head_bytes_sha256=HEAD,
                  derived_full_inventory_matches=True, nonhead_variables_equal_original_initializer=29,
                  original_same_instance_snapshot_available=False, original_batches=50,
                  independent_numpy_loss_derivatives=True, cpu_encoder_autodiff=True,
                  optimizer_updates=0, all_model_and_bn_variables_unchanged=True,
                  traces=traces,
                  largest_loss_absolute_difference_from_original_gpu={
                      key: max(abs(r["loss_difference_from_original_gpu"][key]) for r in traces)
                      for key in ("main_bce", "pair_bce", "consistency")},
                  largest_gradient_relative_l2_difference_from_original_gpu={
                      key: max(r["gradient_comparisons"][key]["relative_l2_error"] for r in traces)
                      for key in references},
                  mac_median_supervised_encoder_norm=float(np.median(ns)),
                  mac_median_consistency_encoder_norm=float(np.median(nd)), mac_R=ratio,
                  mac_fixed_rule_coefficient=float(coefficient), original_fixed_coefficient=original["proposed_fixed_coefficient"],
                  gradients_sha256=sha((args.output / "mac-gradients.npz").read_bytes()),
                  tensorflow_version=tf.__version__, numpy_version=np.__version__,
                  upstream_revision=UPSTREAM, upstream_source_sha256=source_hashes,
                  script_sha256=sha(Path(__file__).read_bytes()),
                  numpy_derivative_source_sha256=sha(Path(__file__).with_name(
                      "probe_rendering_consistency_gradient_mac.py").read_bytes()),
                  limits="Derived numerical state, not recovered historical instance. Original GPU versus "
                         "Mac CPU float gradient differences are descriptive, not a universal equivalence "
                         "or passed-smoke claim. No original probe restart, coefficient revision, parameter "
                         "update, development scoring, hardware operation or model acceptance.")
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in (
        "largest_loss_absolute_difference_from_original_gpu",
        "largest_gradient_relative_l2_difference_from_original_gpu", "mac_R", "mac_fixed_rule_coefficient")}, indent=2))


if __name__ == "__main__":
    main()
