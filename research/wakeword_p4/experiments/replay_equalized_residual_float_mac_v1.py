"""Reconstruct both saved residual float baselines on the native Mac backend."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(ref, payload, upstream, out):
    import numpy as np
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    ref = subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    sources = {}
    def raw(path):
        data = subprocess.check_output(["git", "show", f"{ref}:{path}"])
        sources[path] = hashlib.sha256(data).hexdigest()
        return data
    root = "research/wakeword_p4/"
    locked = json.loads(raw(root+"results/4090-hey-broad-equalized-screen-execution-v1.json"))
    needle = ".local/wakeword/microWakeWord/"
    verified = {}
    for path, expected in locked["files"].items():
        if needle in path and path.endswith(".py"):
            relative = path.split(needle, 1)[1]
            assert sha(upstream/relative) == expected, relative
            verified[relative] = expected
    assert len(verified) == 20
    for name, version in locked["versions"].items():
        assert importlib.metadata.version(name) == version, name
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    namespace = dict(argparse=argparse, mixednet=mixednet, np=np)
    # Execute only the previously inspected pure architecture and head helpers,
    # not their GPU lock/run CLIs or module-level filesystem setup.
    architecture = ast.parse(raw(root+"experiments/probe_hey_context_transfer_shapes_v1.py"))
    head = ast.parse(raw(root+"experiments/probe_hey_broad_bounded_float_v1.py"))
    nodes = [n for n in architecture.body if
             isinstance(n, ast.FunctionDef) and n.name == "create" or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARCH" for t in n.targets)]
    nodes += [n for n in head.body if isinstance(n, ast.FunctionDef) and n.name == "bounded"]
    assert len(nodes) == 3
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<bound-gpu-architecture-helpers>", "exec"), namespace)
    screen = json.loads(raw(root+"results/4090-hey-broad-equalized-screen-result-v1.json"))
    receipt = json.loads(raw(root+"results/4090-hey-broad-equalized-weights-receipt-v1.json"))
    out.mkdir(exist_ok=False)
    cases = []
    for target, order in [("hey_willow", 14), ("hey_nyx", 48394)]:
        weights, = [r for r in receipt["arms"] if r["target"] == target]
        scored, = [r for r in screen["cases"] if r["target"] == target and r["request_order"] == order]
        assert sha(payload/weights["path"]) == weights["sha256"]
        assert sha(payload/scored["arrays_path"]) == scored["arrays_sha256"]
        with np.load(payload/scored["arrays_path"], allow_pickle=False) as saved:
            features, baseline = saved["raw"].copy(), saved["float"].copy()
            expected_phase = saved["offline_phase"].copy()
        values = np.clip(np.pad(features, ((0, (-len(features)) % 3), (0, 0))).astype(np.float32)/25.6, 0, 26)
        models, snapshots, originals = [], [], []
        for _ in range(2):
            model = namespace["create"](225)
            model.load_weights(payload/weights["path"])
            for layer in model.layers:
                cell = getattr(layer, "cell", layer)
                if isinstance(cell, tf.keras.layers.BatchNormalization):
                    layer.trainable = cell.trainable = False
            originals.append(model)
            snapshots.append([v.numpy().copy() for v in model.variables])
            models.append(namespace["bounded"](model, tf))
        stream = utils.to_streaming_inference(models[0], {"stride": 3, "spectrogram_length": 225}, modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
        stream(np.zeros((1, 3, 40), np.float32), training=False)
        states = [v for v in stream.variables if "states" in v.name]
        assert len(states) == 6
        infer = tf.function(lambda x: stream(x, training=False), input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
        window = tf.function(lambda x: models[1](x, training=False), input_signature=[tf.TensorSpec([1, 225, 40], tf.float32)])
        repeats = []
        for _ in range(2):
            for state in states:
                state.assign(tf.zeros_like(state))
            repeats.append(np.asarray([float(infer(values[None, at:at+3]).numpy()[0, 0]) for at in range(0, len(values), 3)]))
        assert np.array_equal(*repeats)
        error = float(np.max(np.abs(repeats[0]-baseline)))
        assert error <= 1e-4
        phase = np.asarray([float(window(np.concatenate((values[end-224:end], np.zeros((1, 40), np.float32)))[None]).numpy()[0, 0])
                            for end in range(225, len(values)+1, 3)])
        phase_error = float(np.max(np.abs(phase-repeats[0][74:])))
        gpu_phase_error = float(np.max(np.abs(phase-expected_phase)))
        assert phase_error <= 1e-4 and gpu_phase_error <= 1e-4
        for model, snapshot in zip(originals, snapshots):
            assert all(np.array_equal(v.numpy(), before) for v, before in zip(model.variables, snapshot, strict=True))
        filename = target+".npz"
        np.savez_compressed(out/filename, probability=repeats[0], repeat=repeats[1], gpu_baseline=baseline, offline_phase=phase)
        case = dict(target=target, request_order=order, steps=len(baseline), twice_reset_exact=True,
                    mac_vs_saved_gpu_float_max_abs=error, actual_offline_phase_max_abs=phase_error,
                    mac_vs_saved_gpu_offline_phase_max_abs=gpu_phase_error,
                    original_variables_unchanged=True, arrays=filename, arrays_sha256=sha(out/filename))
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = dict(schema="mac-equalized-residual-float-replay-v1", gpu_ref=ref,
                  source_artifacts=sources, upstream_python_sha256=verified,
                  versions={n: importlib.metadata.version(n) for n in ["tensorflow", "numpy", "keras", "h5py"]},
                  script_sha256=sha(__file__), cases=cases,
                  limits="Two fixed training failures only. Actual equalized H5, original streaming conversion and bounded head execute on Mac CPU2threads. No optimizer, export, recognition/FAR, new development/acceptance, waveform playback/recording or board write. Does not establish all270 or candidate/P4 admission.")
    with (out/"report.json").open("x") as f:
        f.write(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu-ref", required=True)
    p.add_argument("--payload-root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    replay(a.gpu_ref, a.payload_root, a.upstream, a.output_dir)
