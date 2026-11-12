"""Measure the rejected original v46 on every balanced training factorial cell.

No fitting, threshold selection, new development/final data, audio hardware or
board access. Full classifier inputs are retained, without temporal pooling.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time

import numpy as np

from balanced_representation import (
    ARMS, PHRASES, across_ids, grouped_records, group_measurements, require,
)


UPSTREAM = "4665173cd35f1cff9a61e06fc427f124766c488e"
PACK_REPORT = "7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44"
TRAINING = "bfa9cb5995c28987fc1e0f7b1b53382ce273efed6979b795122cca6df1a33858"
MODEL = "5ac62943e862b2a8b7c92d97b1b4e89a096a2bb9d76f27b1b4b72b25a427f417"
WEIGHTS = "16b01e2f17473121c635801e324f5cf550c2ea0b72f8a054c04d845a0f596027"
VERSIONS = {"tensorflow": "2.21.0", "keras": "3.15.1", "numpy": "2.5.4", "h5py": "3.14.0"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def feature_path(root, row):
    root = root.resolve()
    relative = PurePosixPath(row["raw"])
    require(not relative.is_absolute() and relative.parts[0] == ".."
            and len(relative.parts) == 4
            and relative.parts[1] in ("piper-balanced-training-features-v1",
                                      "balanced-rendering-factorial-v1")
            and relative.parts[2] == "artifacts"
            and relative.parts[3].endswith("-raw.npy"), "Unexpected raw feature path")
    resolved = (root / Path(*relative.parts[1:])).resolve()
    require(resolved.is_relative_to(root), "Raw feature escapes input root")
    return resolved


def load_raw(path, row, authenticate=False):
    if authenticate:
        data = path.read_bytes()
        require(sha(data) == row["raw_file_sha256"], "Original NPY bytes differ: " + str(path))
    raw = np.load(path, allow_pickle=False)
    require(raw.dtype == np.dtype("<u2") and list(raw.shape) == row["raw_shape"],
            "Raw dtype or shape differs")
    if authenticate:
        require(sha(raw.tobytes()) == row["raw_sha256"], "Original array bytes differ")
    return raw


def inventory(variables):
    # Keras variables use a dtype string, whereas loaded SavedModel variables
    # expose a tf.DType. Compare the actual numeric storage type and bytes.
    values = [v.numpy() for v in variables]
    return Counter((value.shape, value.dtype.str, sha(value.tobytes())) for value in values)


def saved_model_preflight(path, window_infer, groups, feature_root, tf, model):
    """Compare original-export state/weights and all60 first-ID training traces.

    This finite preflight does not claim whole-domain SavedModel equivalence.
    Every source in the main diagnostic is still measured with the H5 graph.
    """
    if path is None:
        return dict(performed=False,
                    limits="Original SavedModel omitted; H5 graph inference only, no export parity claim")
    files = {p.relative_to(path).as_posix(): sha(p.read_bytes())
             for p in sorted(path.rglob("*")) if p.is_file()}
    require(bool(files), "Empty original SavedModel")
    loaded = tf.saved_model.load(str(path))
    signature = loaded.signatures["serving_default"]
    inputs = signature.structured_input_signature[1]
    require(len(inputs) == 1 and tuple(next(iter(inputs.values())).shape) in ((1, 3, 40), (None, 3, 40)),
            "Unexpected SavedModel input")
    key = next(iter(inputs))
    states = [v for v in loaded.variables if "states" in v.name]
    nonstates = [v for v in loaded.variables if "states" not in v.name]
    require(len(states) == 6 and inventory(nonstates) == inventory(model.variables),
            "Original SavedModel nonstate variable inventory differs from H5")
    before = inventory(nonstates)
    results = []
    first_id = groups[0][0]
    for voice, speed, rows, ends in groups:
        if voice != first_id:
            break
        for row in rows:
            raw = load_raw(feature_path(feature_root, row), row)
            values = np.clip(raw.astype(np.float32) / 25.6, 0, 26)

            def stream():
                for state in states:
                    state.assign(tf.zeros_like(state))
                scores = []
                for at in range(0, len(values) - 2, 3):
                    result = signature(**{key: tf.convert_to_tensor(values[None, at:at + 3])})
                    require(len(result) == 1, "Unexpected SavedModel outputs")
                    scores.append(float(next(iter(result.values())).numpy().reshape(-1)[0]))
                return np.asarray(scores, dtype=np.float32)

            scores = stream()
            if not results:
                np.testing.assert_array_equal(scores, stream())
            windows = np.stack([values[end - 149:end + 1] for end in ends])
            _, reference = window_infer(windows)
            reference = reference.numpy().reshape(-1)
            observed = scores[ends // 3 - 1]
            np.testing.assert_allclose(reference, observed, atol=2e-5, rtol=2e-5)
            results.append(dict(source=row["source"], arm=row["arm"], speed=speed,
                                compared_windows=len(ends), stream_steps=len(scores),
                                maximum_probability_error=float(np.max(np.abs(reference - observed))),
                                stream_probability_range=float(np.ptp(scores))))
    require(len(results) == 60 and inventory(nonstates) == before,
            "Incomplete first-ID census or changed nonstate variable")
    require(any(r["stream_probability_range"] > 1e-6 for r in results),
            "All original-export preflight traces are uninformative constants")
    return dict(performed=True, saved_model_current_file_sha256=files,
                nonstate_variable_inventory_matches_h5=True, state_buffers=6,
                first_trace_reset_exact=True, first_numeric_conditioning_id=first_id,
                traces=results, all60_predeclared_training_traces_match=True,
                limits="Finite first-ID/three-speed/four-phrase/five-arm preflight, not universal "
                       "export equivalence. Current received SavedModel hashes are reported; "
                       "historical export record did not bind these per-file hashes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("records", "feature-root", "model-dir", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--saved-model", type=Path,
                        help="Original received v46 export only; never regenerate to fill this input")
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh diagnostic output directory required")
    feature_root, upstream = args.feature_root.resolve(), args.upstream.resolve()
    raw_report = args.records.read_bytes()
    require(sha(raw_report) == PACK_REPORT, "Original40320 report differs")
    report = json.loads(raw_report)
    records = report["records"]
    require(report["included"] == len(records) == 40320 and not report["final_data_used"],
            "Incomplete or wrong source report")
    voices = [str(v) for v in np.random.default_rng(726).permutation(904)[:672]]
    groups = grouped_records(records, voices)
    require(len(groups) == 2016, "Incomplete group census")
    model_dir = args.model_dir.resolve()
    training_bytes = (model_dir / "training.json").read_bytes()
    require(sha(training_bytes) == TRAINING
            and sha((model_dir / "hey_nyx.tflite").read_bytes()) == MODEL
            and sha((model_dir / "best.weights.h5").read_bytes()) == WEIGHTS,
            "Original rejected v46 model artifacts differ")
    training = json.loads(training_bytes)
    require(training["model_sha256"] == MODEL and training["selected_step"] == 10000
            and training["context_frames"] == 150 and training["architecture"] == "teacher60"
            and training["validation_window_phase"] == 1, "Original model recipe differs")
    require(git(upstream, "rev-parse", "HEAD").decode().strip() == UPSTREAM,
            "Unexpected local upstream revision")
    upstream_sources = {}
    for name in git(upstream, "ls-files", "microwakeword").decode().splitlines():
        if name.endswith(".py"):
            data = (upstream / name).read_bytes()
            require(data == git(upstream, "show", UPSTREAM + ":" + name),
                    "Modified local upstream: " + name)
            upstream_sources[name] = sha(data)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    require(versions == VERSIONS, "Use the tested Mac float-runtime versions")
    # Authenticate every original file/array before any candidate inference.
    paths, receipt = set(), hashlib.sha256()
    for i, row in enumerate(records):
        path = feature_path(feature_root, row)
        require(path not in paths, "Duplicate raw feature path")
        paths.add(path)
        load_raw(path, row, authenticate=True)
        receipt.update((row["raw"] + "\t" + row["raw_file_sha256"] + "\t"
                        + row["raw_sha256"] + "\n").encode())
        if (i + 1) % 4000 == 0:
            print(json.dumps(dict(phase="authenticate_raw", completed=i + 1, total=40320)), flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet
    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    model = mixednet.model(architecture.parse_args(training["architecture_args"]), (150, 40), None)
    model.load_weights(model_dir / "best.weights.h5")
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    require(model.count_params() == 22801, "Unexpected parameter count")
    head = model.layers[-1]
    require(isinstance(head, tf.keras.layers.Dense) and tuple(head.input.shape) == (None, 300),
            "Expected complete flattened five-by-sixty classifier input")
    require(all(np.isfinite(v.numpy()).all() for v in model.variables), "Nonfinite original weights")
    weights_before = inventory(model.variables)
    classifier, bias = head.get_weights()
    classifier, bias = classifier[:, 0], float(bias[0])
    probe = tf.keras.Model(model.input, [head.input, model.output])

    @tf.function(input_signature=[tf.TensorSpec([None, 150, 40], tf.float32)])
    def infer(x):
        return probe(x, training=False)

    preflight = saved_model_preflight(args.saved_model, infer, groups, feature_root, tf, model)
    args.output.mkdir(parents=True)
    (args.output / "factorized-original-report.json").write_bytes(raw_report)
    (args.output / "training-original.json").write_bytes(training_bytes)
    np.savez_compressed(args.output / "classifier.npz", weight=classifier, bias=np.array(bias))
    details, started = [], time.monotonic()
    for index, (voice, speed, rows, ends) in enumerate(groups):
        windows = []
        for row in rows:
            raw = load_raw(feature_path(feature_root, row), row)
            values = np.clip(raw.astype(np.float32) / 25.6, 0, 26)
            windows.extend(values[end - 149:end + 1] for end in ends)
        embeddings, probabilities = [v.numpy() for v in infer(np.stack(windows))]
        embeddings = embeddings.reshape(4, 5, len(ends), 300)
        probabilities = probabilities.reshape(4, 5, len(ends))
        measurements = group_measurements(embeddings, probabilities, classifier, bias)
        name = "group-" + str(index) + ".npz"
        np.savez_compressed(args.output / name, ends=ends, embeddings=embeddings,
                            probabilities=probabilities)
        details.append(dict(voice=voice, speed=speed, windows=len(ends),
                            pack_indices=[r["pack_index"] for r in rows],
                            arrays=name, arrays_sha256=sha((args.output / name).read_bytes()),
                            measurements=measurements))
        if (index + 1) % 48 == 0:
            print(json.dumps(dict(phase="representations", completed=index + 1, total=2016,
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    require(inventory(model.variables) == weights_before, "Frozen inference changed model variables")
    imports = {}
    for name, module in sorted(sys.modules.items()):
        if name.startswith("microwakeword") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            require(path.is_relative_to(upstream), "Imported wrong upstream module")
            relative = path.relative_to(upstream).as_posix()
            require(relative in upstream_sources, "Imported untracked upstream source")
            imports[name] = upstream_sources[relative]
    detail_bytes = (json.dumps(details, separators=(",", ":"), allow_nan=False) + "\n").encode()
    (args.output / "groups.json").write_bytes(detail_bytes)
    result = dict(
        scope=__doc__, accepted=False, model_sha256=MODEL, weights_sha256=WEIGHTS,
        training_sha256=TRAINING, source_report_sha256=PACK_REPORT,
        authenticated_raw_arrays=40320, raw_identity_receipt_sha256=receipt.hexdigest(),
        conditioning_ids=672, groups=2016, sources=8064, phrase_order=list(PHRASES),
        arm_order=list(ARMS), common_absolute_windows=sum(len(g[3]) for g in groups),
        measured_classifier_inputs=sum(len(g[3]) * 20 for g in groups),
        representation_shape=[300], phase=1,
        weighting="Mean of defined windows within each ID/speed/phrase/arm comparison; "
                  "mean three speed means within ID, then equal conditioning IDs. "
                  "Undefined metric counts and partial groups retained; no confidence intervals.",
        metric_direction="Nuisance left=quiet/right=other arm; near-word left=Hey Nyx/right=near word. "
                         "logit_delta is right minus left, not a recognition event.",
        groups_sha256=sha(detail_bytes), classifier_npz_sha256=sha((args.output / "classifier.npz").read_bytes()),
        summaries=across_ids(details), original_saved_model_preflight=preflight,
        upstream_revision=UPSTREAM, upstream_source_sha256=upstream_sources,
        imported_upstream_module_sha256=imports, versions=versions,
        python=platform.python_version(), platform=platform.platform(),
        weights_unchanged=True, script_sha256=sha(Path(__file__).read_bytes()),
        metrics_source_sha256=sha(Path(__file__).with_name("balanced_representation.py").read_bytes()),
        seconds=time.monotonic() - started,
        limits=[
            "Frozen selected10000 v46 original weights; training-only conditioning grid already examined. "
            "No generalization, independent-speaker or causal explanation of development errors.",
            "Quiet/noise/RIR/combined conditions retain original gains, noise and tails. "
            "Common absolute positions avoid selecting different peaks; this does not isolate every rendering factor.",
            "Complete generated containers and stored source boundaries are not verified phonetic endpoints.",
            "The 300-dimensional Dense input retains five ordered positions. No extra embedding loss, "
            "temporal reduction, fitting, threshold choice or INT8/new-development/final/P4/audio operation.",
            "Local clean upstream bytes are authenticated. GPU historical upstream bytes require a separate "
            "receipt; any finite original-export preflight is reported at its actual scope."])
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("authenticated_raw_arrays", "conditioning_ids", "groups",
                                            "measured_classifier_inputs", "seconds", "accepted")}, indent=2))


if __name__ == "__main__":
    main()
