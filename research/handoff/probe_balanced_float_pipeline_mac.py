"""Test balanced float extraction against an original-format random export.

All inputs and weights are synthetic/local. No v46/v47 model, original corpus,
development/final sample, audio device or development board is accessed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from balanced_representation import ARMS, PHRASES, SPEEDS, across_ids, grouped_records, group_measurements, require
from measure_balanced_float_representation import UPSTREAM, load_raw, saved_model_preflight


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required")
    upstream = args.upstream.resolve()
    require(subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"]).decode().strip()
            == UPSTREAM, "Unexpected upstream revision")
    source_hashes = {}
    tracked = subprocess.check_output(["git", "-C", str(upstream), "ls-files", "microwakeword"]).decode()
    for name in tracked.splitlines():
        if name.endswith(".py"):
            data = (upstream / name).read_bytes()
            require(data == subprocess.check_output(["git", "-C", str(upstream), "show", UPSTREAM + ":" + name]),
                    "Modified upstream source")
            source_hashes[name] = sha(data)
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(726)
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    from probe_mww_float_runtime_mac import RECIPE_PATH, RECIPE_REF, ROOT
    recipe_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show", RECIPE_REF + ":" + RECIPE_PATH])
    recipe = json.loads(recipe_bytes)
    p = argparse.ArgumentParser()
    mixednet.model_parameters(p)
    flags = p.parse_args(recipe["architecture_args"])
    model = mixednet.model(flags, (150, 40), None)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    require(model.count_params() == 22801, "Unexpected random graph geometry")
    probe = tf.keras.Model(model.input, [model.layers[-1].input, model.output])
    classifier, bias = model.layers[-1].get_weights()

    @tf.function(input_signature=[tf.TensorSpec([None, 150, 40], tf.float32)])
    def infer(x):
        return probe(x, training=False)

    stream_source = mixednet.model(flags, (150, 40), None)
    for layer in stream_source.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    stream_source.set_weights(model.get_weights())
    streaming = utils.to_streaming_inference(stream_source,
        {"stride": 3, "spectrogram_length": 150}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    streaming(np.zeros((1, 3, 40), np.float32))
    rng = np.random.default_rng(726)
    with tempfile.TemporaryDirectory(prefix="nyx-balanced-float-smoke-") as scratch:
        root = Path(scratch)
        streaming.export(str(root / "random-saved-model"), verbose=False)
        artifacts = root / "piper-balanced-training-features-v1/artifacts"
        artifacts.mkdir(parents=True)
        rows = []
        for speed in SPEEDS:
            for phrase in PHRASES:
                for arm in ARMS:
                    length = 361 if arm in ("combined", "rir") else 355
                    raw = rng.integers(0, 667, (length, 40), dtype=np.uint16)
                    name = str(len(rows)) + "-raw.npy"
                    path = artifacts / name
                    np.save(path, raw)
                    rows.append(dict(voice="0", length_multiplier=speed, target=phrase,
                        arm=arm, split="training", mechanical_pass=True, entire_signal_fits=True,
                        source=str(len(rows)) + ".synthetic", raw_shape=list(raw.shape),
                        feature_crop_end=262 if length == 361 else 256,
                        raw="../piper-balanced-training-features-v1/artifacts/" + name,
                        raw_file_sha256=sha(path.read_bytes()), raw_sha256=sha(raw.tobytes())))
                    load_raw(path, rows[-1], authenticate=True)
        groups = grouped_records(rows, ["0"])
        preflight = saved_model_preflight(root / "random-saved-model", infer, groups, root, tf, model)
        details = []
        for voice, speed, records, ends in groups:
            windows = []
            for row in records:
                raw = np.load(artifacts / Path(row["raw"]).name, allow_pickle=False)
                values = np.clip(raw.astype(np.float32) / 25.6, 0, 26)
                windows.extend(values[e - 149:e + 1] for e in ends)
            embeddings, probabilities = [v.numpy() for v in infer(np.stack(windows))]
            measurements = group_measurements(embeddings.reshape(4, 5, len(ends), 300),
                probabilities.reshape(4, 5, len(ends)), classifier[:, 0], float(bias[0]))
            details.append(dict(voice=voice, speed=speed, measurements=measurements))
        summary = across_ids(details)
        require(len(summary) == 257 and all(r["total_conditioning_ids"] == 1 for r in summary.values()),
                "Incomplete diagnostic measurement set")
    report = dict(scope=__doc__, random_traces=60, synthetic_groups=3,
                  all_raw_hash_checks_pass=True, original_format_saved_model_preflight=preflight,
                  descriptive_measurements=257, all_checks_pass=True,
                  upstream_revision=UPSTREAM, upstream_source_sha256=source_hashes,
                  recipe_sha256=sha(recipe_bytes), versions={n:importlib.metadata.version(n)
                    for n in ("tensorflow", "keras", "numpy", "h5py")},
                  script_sha256=sha(Path(__file__).read_bytes()),
                  helper_sha256={name:sha(Path(__file__).with_name(name).read_bytes()) for name in
                    ("balanced_representation.py", "measure_balanced_float_representation.py")},
                  original_candidate_or_corpus_accessed=False, accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in
                    ("original_format_saved_model_preflight", "upstream_source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
