"""Exercise the pinned float inference graph using synthetic features only.

No received candidate, training feature, development/final data, microphone or
device is accessed. This prepares the Mac runtime for a later original-artifact
diagnostic; random weights cannot establish candidate or MCU performance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf


UPSTREAM = "4665173cd35f1cff9a61e06fc427f124766c488e"
RECIPE_REF = "65ebbcce0c87bb8a51cb5abfcd56a1c9bfe0ab4f"
RECIPE_PATH = "research/wakeword_p4/results/4090-v46-training.json"
ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh report required")
    upstream = args.upstream.resolve()
    require(git(upstream, "rev-parse", "HEAD").decode().strip() == UPSTREAM,
            "Unexpected microWakeWord revision")
    source_hashes = {}
    for name in git(upstream, "ls-files", "microwakeword").decode().splitlines():
        if not name.endswith(".py"):
            continue
        data = (upstream / name).read_bytes()
        require(data == git(upstream, "show", UPSTREAM + ":" + name),
                "Modified upstream source: " + name)
        source_hashes[name] = sha(data)
    recipe_bytes = git(ROOT, "show", RECIPE_REF + ":" + RECIPE_PATH)
    recipe = json.loads(recipe_bytes)
    require(recipe["architecture"] == "teacher60"
            and recipe["context_frames"] == 150
            and recipe["validation_window_phase"] == 1
            and recipe["parameters"] == 22801, "Unexpected recipe architecture")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes

    architecture = argparse.ArgumentParser()
    mixednet.model_parameters(architecture)
    flags = architecture.parse_args(recipe["architecture_args"])

    def build():
        model = mixednet.model(flags, (150, 40), None)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        require(model.count_params() == 22801, "Unexpected parameter count")
        return model

    window_model = build()
    head = window_model.layers[-1]
    require(isinstance(head, tf.keras.layers.Dense)
            and tuple(head.input.shape) == (None, 300),
            "Expected flattened five-by-sixty penultimate representation")
    window_probe = tf.keras.Model(window_model.input,
                                  [head.input, window_model.output])
    initial_weights = [w.copy() for w in window_model.get_weights()]
    initial_weight_hash = sha(b"".join(w.tobytes() for w in initial_weights))
    # Keras3 H5 compatibility is checked with our random weights. These are not
    # original v46 weights, and nothing is saved as a candidate or exported.
    stream_source = build()
    with tempfile.TemporaryDirectory(prefix="nyx-float-runtime-") as scratch:
        path = Path(scratch) / "random.weights.h5"
        window_model.save_weights(path)
        random_h5_sha = sha(path.read_bytes())
        stream_source.load_weights(path)
        for before, after in zip(initial_weights, stream_source.get_weights()):
            np.testing.assert_array_equal(before, after)
    # Conversion changes mode attributes on its source, so use a separate model
    # and keep the untouched non-stream graph for the window comparison.
    streaming = utils.to_streaming_inference(
        stream_source, {"stride": 3, "spectrogram_length": 150},
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), dtype=np.float32), training=False)
    stream_head = streaming.layers[-1]
    stream_probe = tf.keras.Model(streaming.input,
                                  [stream_head.input, streaming.output])
    states = [v for v in stream_probe.variables if "states" in v.name]
    require(len(states) == 6, "Unexpected streaming buffer count")
    nonstate = [v for v in stream_probe.variables if "states" not in v.name]
    frozen = [v.numpy().copy() for v in nonstate]

    @tf.function(input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
    def infer(x):
        return stream_probe(x, training=False)

    def replay(raw):
        for state in states:
            state.assign(tf.zeros_like(state))
        values = np.clip(raw.astype(np.float32) / 25.6, 0, 26)
        embeddings, scores = [], []
        for at in range(0, len(values) - 2, 3):
            embedding, score = infer(values[None, at:at + 3])
            embeddings.append(embedding.numpy()[0])
            scores.append(float(score.numpy()[0, 0]))
        return np.stack(embeddings), np.asarray(scores, dtype=np.float32)

    rng = np.random.default_rng(726)
    inputs = [
        ("zeros", np.zeros((450, 40), dtype=np.uint16)),
        ("random_feature_range", rng.integers(0, 667, (604, 40), dtype=np.uint16)),
        ("raw_uint16_extremes", rng.integers(0, 65536, (452, 40), dtype=np.uint16)),
    ]
    results = []
    for name, raw in inputs:
        embedding, score = replay(raw)
        # Interpose a distinct stream to verify reset across clip boundaries.
        replay(np.flip(inputs[1][1], axis=0))
        repeated_embedding, repeated_score = replay(raw)
        np.testing.assert_array_equal(repeated_embedding, embedding)
        np.testing.assert_array_equal(repeated_score, score)
        require(np.isfinite(embedding).all() and np.isfinite(score).all(),
                "Nonfinite streaming output")
        values = np.clip(raw.astype(np.float32) / 25.6, 0, 26)
        # This phase is fixed by the existing recipe, never searched here.
        phase = (150 - flags.first_conv_kernel_size) % flags.stride
        require(phase == recipe["validation_window_phase"] == 1,
                "Known full-window/stream phase changed")
        ends = np.arange(150, len(values) - phase + 1, 3)
        windows = np.stack([values[end - 150 + phase:end + phase] for end in ends])
        reference_embedding, reference_score = [v.numpy() for v in
                                                window_probe(windows, training=False)]
        mature_embedding, mature_score = embedding[ends // 3 - 1], score[ends // 3 - 1]
        np.testing.assert_allclose(reference_embedding, mature_embedding, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(reference_score[:, 0], mature_score, rtol=2e-5, atol=2e-6)
        # Report phase0 sensitivity without selecting it or treating either
        # phase as a candidate quality result.
        phase0 = np.stack([values[end - 150:end] for end in ends])
        unaligned_embedding = window_probe(phase0, training=False)[0].numpy()
        results.append(dict(
            name=name, raw_shape=list(raw.shape), raw_sha256=sha(raw.tobytes()),
            steps=len(score), unused_tail_frames=len(raw) % 3,
            compared_mature_windows=len(ends), known_window_phase=phase,
            maximum_embedding_error=float(np.max(np.abs(reference_embedding - mature_embedding))),
            maximum_probability_error=float(np.max(np.abs(reference_score[:, 0] - mature_score))),
            phase0_maximum_embedding_error=float(np.max(np.abs(unaligned_embedding - mature_embedding))),
            all_reset_outputs_exact=True,
            embedding_sha256=sha(embedding.tobytes()), score_sha256=sha(score.tobytes())))
    for before, after in zip(frozen, nonstate):
        np.testing.assert_array_equal(before, after.numpy())
    for before, after in zip(initial_weights, window_model.get_weights()):
        np.testing.assert_array_equal(before, after)
    versions = {name: importlib.metadata.version(name) for name in (
        "tensorflow", "keras", "numpy", "h5py", "ai-edge-litert", "pymicro-features")}
    imported = {}
    for name, module in sorted(sys.modules.items()):
        if not name.startswith("microwakeword") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(upstream), "Imported wrong microWakeWord: " + name)
        relative = path.relative_to(upstream).as_posix()
        require(relative in source_hashes, "Untracked imported upstream module: " + relative)
        imported[name] = dict(path=relative, sha256=source_hashes[relative])
    report = dict(
        scope=__doc__, platform=platform.platform(), python=platform.python_version(),
        versions=versions, upstream_revision=UPSTREAM,
        upstream_python_sha256=source_hashes, imported_upstream_modules=imported,
        recipe_ref=RECIPE_REF, recipe_path=RECIPE_PATH, recipe_sha256=sha(recipe_bytes),
        architecture_args=recipe["architecture_args"], context_frames=150, parameters=22801,
        weight_source="Locally seeded random initialization only; no candidate weights loaded",
        initial_random_weight_sha256=initial_weight_hash, temporary_random_h5_sha256=random_h5_sha,
        random_h5_roundtrip_exact=True, penultimate_flattened_shape=[300],
        state_buffers=[dict(name=v.name, path=getattr(v, "path", v.name),
                            shape=list(v.shape)) for v in states],
        nonstate_weights_unchanged=True, results=results, all_checks_pass=True,
        script_sha256=sha(Path(__file__).read_bytes()), accepted=False,
        original_v46_model_received_or_inferred=False,
        training_or_development_or_final_data_accessed=False,
        limits="Local clean pinned-upstream graph and synthetic-input runtime compatibility only. "
               "GPU upstream working-tree bytes are not authenticated by its training record. "
               "Random H5 roundtrip does not prove original candidate H5 compatibility. "
               "No original SavedModel, weights, features, INT8, recognition, FAR, Hi ESP, "
               "P4, audio playback/capture or promotion evidence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("upstream_python_sha256", "imported_upstream_modules")}, indent=2))


if __name__ == "__main__":
    main()
