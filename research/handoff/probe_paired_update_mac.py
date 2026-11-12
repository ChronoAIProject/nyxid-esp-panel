"""Check the pinned training-step method against an independent Adam first step.

Use certified numerical initialization and the first actual saved batch. The
oracle gradients come from independent NumPy loss derivatives plus CPU encoder
autodiff, not the helper's losses. This method-only check does not instantiate
or certify its sampler/constructor or prove whole-training equivalence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REF = "ef6864aa5bdbb701383b69ac78ba90e003a0ae83"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("probe", "initialization", "replay", "v46-model", "upstream", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Fresh output required"
    repo = Path(__file__).resolve().parents[2]
    replay_bytes = (args.replay / "report.json").read_bytes()
    replay = json.loads(replay_bytes)
    assert replay["original_probe_sha256"] == "0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f"
    assert replay["script_sha256"] == sha(Path(__file__).with_name("replay_paired_init_gradients_mac.py").read_bytes())
    assert replay["derived_h5_sha256"] == sha((args.initialization / "derived-probe-initialization.weights.h5").read_bytes())
    assert replay["gradients_sha256"] == sha((args.replay / "mac-gradients.npz").read_bytes())
    probe = json.loads((args.probe / "gradient-probe.json").read_bytes())
    assert sha((args.probe / "probe-batches.npz").read_bytes()) == probe["batches_sha256"]
    with np.load(args.replay / "mac-gradients.npz", allow_pickle=False) as data:
        gs, gd = data["supervised"][0].astype(np.float64), data["consistency"][0].astype(np.float64)
    with np.load(args.probe / "probe-batches.npz", allow_pickle=False) as data:
        inputs = [data[k][0] for k in ("main_x", "main_y", "main_weight", "pair_x", "pair_y", "pair_weight")]
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
    training = json.loads((args.v46_model / "training.json").read_bytes())
    options = argparse.ArgumentParser()
    mixednet.model_parameters(options)
    model = mixednet.model(options.parse_args(training["architecture_args"]), (150, 40), None)
    model.load_weights(args.initialization / "derived-probe-initialization.weights.h5")
    inventory = json.loads((args.initialization / "variable-inventory.json").read_bytes())
    assert [dict(path=v.path, shape=list(v.shape), dtype=v.numpy().dtype.name,
                 value_sha256=sha(v.numpy().tobytes())) for v in model.variables] == inventory
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    head = model.layers[-1]
    extractor = tf.keras.Model(model.input, head.input)
    head_ids = {id(v) for v in head.trainable_variables}
    variables = model.trainable_variables
    assert [(v.path, list(v.shape)) for v in variables] == [
        (v["path"], v["shape"]) for v in probe["variable_layout"]]
    all_before = [v.numpy().copy() for v in model.variables]
    before = [v.numpy().copy() for v in variables]
    sources = {}
    source_directory = tempfile.TemporaryDirectory(prefix="mac-paired-pinned-step-")
    for name in ("paired_rendering_sampler", "rendering_consistency", "paired_rendering_training"):
        path = "research/wakeword_p4/training/" + name + ".py"
        source = subprocess.check_output(["git", "-C", str(repo), "show", REF + ":" + path])
        sources[name] = sha(source)
        # Give inspect/AutoGraph the exact pinned source file. An exec-only
        # module cannot expose source and otherwise falls back to as-is tracing.
        source_file = Path(source_directory.name) / (name + ".py")
        source_file.write_bytes(source)
        specification = importlib.util.spec_from_file_location(name, source_file)
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        specification.loader.exec_module(module)
    helper = sys.modules["paired_rendering_training"].PairedRenderingTraining
    reports, outcomes, contexts = [], [], []
    for coefficient in (0., 1.):
        for variable, value in zip(model.variables, all_before):
            variable.assign(value)
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, global_clipnorm=5), loss="binary_crossentropy")
        model.optimizer.build(variables)
        # Exercise the actual pinned method with its required real model
        # context, without pretending to test pack/sampler construction here.
        context = helper.__new__(helper)
        context.model, context.head, context.extractor = model, head, extractor
        context.coefficient, context.head_ids = coefficient, head_ids
        context.encoder_indices = [i for i, v in enumerate(variables) if id(v) not in head_ids]
        contexts.append(context)
        combined = gs + coefficient * gd
        norm = np.sqrt(np.sum(combined * combined))
        clipped = combined * min(1., 5. / norm)
        # Keras Adam uses epsilon with the uncorrected second moment:
        # alpha=lr*sqrt(1-beta2)/(1-beta1), m=.1g, v=.001g² at t=1.
        delta = (1e-3 * np.sqrt(.001) / .1) * (.1 * clipped) / (np.sqrt(.001 * clipped ** 2) + 1e-7)
        observed = [float(v) for v in context.step(*inputs)]
        assert int(model.optimizer.iterations) == 1
        first = replay["traces"][0]
        np.testing.assert_allclose(observed[:4], [first["main_bce"] + first["pair_bce"] + coefficient * first["consistency"],
                                                first["main_bce"], first["pair_bce"], first["consistency"]],
                                   rtol=2e-5, atol=2e-5)
        per_variable, at, passed = [], 0, True
        after = [v.numpy().copy() for v in variables]
        outcomes.append(after)
        for variable, previous, current in zip(variables, before, after):
            size = previous.size
            expected = delta[at:at + size].reshape(previous.shape)
            actual = previous.astype(np.float64) - current.astype(np.float64)
            error = np.abs(actual - expected)
            # Fixed before execution: two float32 ULPs of the stored parameter
            # plus2e-6absolute step error, to cover storage/gradient rounding.
            bound = 2 * np.abs(np.spacing(np.abs(previous))).astype(np.float64) + 2e-6
            this_pass = bool(np.all(error <= bound))
            passed &= this_pass
            per_variable.append(dict(path=variable.path, maximum_step_absolute_error=float(error.max()),
                                     maximum_fraction_of_fixed_bound=float((error / bound).max()), passed=this_pass))
            at += size
        trainable_ids = {id(v) for v in variables}
        for variable, previous in zip(model.variables, all_before):
            if id(variable) not in trainable_ids:
                np.testing.assert_array_equal(variable.numpy(), previous)
        reports.append(dict(coefficient=coefficient, optimizer_updates=1,
                            observed_total_main_pair_consistency_encoder_norms=observed,
                            independent_combined_gradient_norm=float(norm),
                            all_nontrainable_bn_values_unchanged=True,
                            per_variable=per_variable, all_steps_within_fixed_bound=passed))
    result = dict(scope=__doc__, helper_commit=REF, source_sha256=sources,
                  exact_pinned_source_files_available_to_autograph=True,
                  reference_replay_report_sha256=sha(replay_bytes), reports=reports,
                  maximum_control_treatment_parameter_difference=float(max(
                      np.max(np.abs(a - b)) for a, b in zip(*outcomes))),
                  all_passed=all(r["all_steps_within_fixed_bound"] for r in reports),
                  script_sha256=sha(Path(__file__).read_bytes()), tensorflow_version=tf.__version__,
                  limits="First actual batch at certified derived initialization on Mac CPU only. "
                         "No constructor/sampler/trace replay, original GPU smoke passage, convergence, "
                         "matched120k run or wake-quality/P4 acceptance is inferred.")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(all_passed=result["all_passed"], maximum_control_treatment_parameter_difference=
                         result["maximum_control_treatment_parameter_difference"])))
    source_directory.cleanup()
    assert result["all_passed"], "Preserved optimizer-method mismatch; inspect the original report"


if __name__ == "__main__":
    main()
