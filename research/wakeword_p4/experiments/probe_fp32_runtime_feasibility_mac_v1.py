"""Candidate-free FP32 export and standalone P4 link; never accesses a board."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

GPU_REF = "f9e773a3b6073beaca42a915045cc6d0007b49dc"
PREFIX = "research/wakeword_p4/"
ARCHITECTURE = PREFIX + "experiments/probe_hey_context_transfer_shapes_v1.py"
LOCK = PREFIX + "results/4090-hey-broad-equalized-screen-execution-v1.json"
SEED = 20261114
STEPS = 256
TOLERANCE = 1e-4


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo, path):
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{GPU_REF}:{path}"])


def tree_identity(root):
    files = {str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(root.rglob("*"))
             if p.is_file() and ".git" not in p.relative_to(root).parts}
    return dict(files=len(files), sha256=sha(json.dumps(files, sort_keys=True).encode()))


def identity(repo, upstream):
    locked = json.loads(git_bytes(repo, LOCK))
    needle = ".local/wakeword/microWakeWord/"
    upstream_files = {p.split(needle, 1)[1]: s for p, s in locked["files"].items()
                      if needle in p and p.endswith(".py")}
    assert len(upstream_files) == 20
    assert all(sha((upstream / p).read_bytes()) == s for p, s in upstream_files.items())
    versions = {k: importlib.metadata.version(k) for k in locked["versions"]}
    assert versions == locked["versions"]
    return dict(script_sha256=sha(Path(__file__).read_bytes()), gpu_ref=GPU_REF,
                published_sources={p: sha(git_bytes(repo, p)) for p in [ARCHITECTURE, LOCK]},
                upstream_python_sha256=upstream_files, versions=versions,
                components={name: tree_identity(repo / "managed_components" / ("espressif__" + name))
                            for name in ["esp-tflite-micro", "esp-nn"]})


def plan(repo, upstream):
    return dict(schema="mac-fp32-runtime-feasibility-plan-v1", identity=identity(repo, upstream),
                model="Original context225 raw-sigmoid MixedNet architecture; seeded random weights only",
                seed=SEED, context_frames=225, stride=3, feature_channels=40,
                export="Frozen BN; original internal-state streaming conversion; explicit six zero states; "
                       "ExportArchive; TFLITE_BUILTINS only; no optimizations/quantization/select-TF ops",
                synthetic_streams=["zeros", "constant26", "ramp257", "xorshift32"],
                steps_per_stream=STEPS, fresh_litert_repeats=2, float_max_abs_tolerance=TOLERANCE,
                build="Isolated full ESP-IDF5.5.4 P4 executable link, CPU360MHz/ilp32f; "
                      "copied unchanged runtime sources plus explicit esp-nn CMake dependency; "
                      "prospective internal arena256KiB+variable64KiB, no board execution",
                limits="No original/candidate weights, audio, training/development/acceptance data, "
                       "recognition/FAR, optimizer, thresholds, board reads/writes or production changes. "
                       "Does not change existing integer gates or qualify a rejected model. "
                       "No actual P4 arena usage, inference latency or panel headroom claim.")


def synthetic(np, condition):
    if condition == 0:
        return np.zeros((STEPS, 1, 3, 40), np.float32)
    if condition == 1:
        return np.full((STEPS, 1, 3, 40), 26, np.float32)
    if condition == 2:
        return (np.arange(STEPS * 120, dtype=np.uint32) % 257).astype(np.float32).reshape(
            STEPS, 1, 3, 40) * np.float32(26 / 256)
    values = []
    state = 0x4e595831
    for _ in range(STEPS * 120):
        state ^= (state << 13) & 0xffffffff
        state ^= state >> 17
        state ^= (state << 5) & 0xffffffff
        values.append(state & 65535)
    return np.asarray(values, np.float32).reshape(STEPS, 1, 3, 40) * np.float32(26 / 65535)


def export(repo, upstream, out, frozen):
    import numpy as np
    import tensorflow as tf
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    from tensorflow.lite.python import schema_py_generated as schema
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    sys.path.insert(0, str(upstream))
    from microwakeword import mixednet, utils
    from microwakeword.layers import modes
    source = ast.parse(git_bytes(repo, ARCHITECTURE))
    nodes = [n for n in source.body if
             (isinstance(n, ast.FunctionDef) and n.name == "create") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ARCH" for t in n.targets))]
    assert len(nodes) == 2
    namespace = dict(argparse=argparse, mixednet=mixednet)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned-architecture-only>", "exec"), namespace)
    original = namespace["create"](225)
    original_snapshot = [v.numpy().copy() for v in original.variables]
    for layer in original.layers:
        cell = getattr(layer, "cell", layer)
        if isinstance(cell, tf.keras.layers.BatchNormalization):
            layer.trainable = cell.trainable = False
    stream = utils.to_streaming_inference(original, {"stride": 3, "spectrogram_length": 225},
                                          modes.Modes.STREAM_INTERNAL_STATE_INFERENCE)
    stream(np.zeros((1, 3, 40), np.float32), training=False)
    states = [v for v in stream.variables if "states" in v.name]
    assert len(states) == 6
    for state in states:
        state.assign(tf.zeros_like(state))
    archive = tf.keras.export.ExportArchive()
    archive.track(stream)
    archive.add_endpoint(name="serve", fn=stream.call,
                         input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
    archive.write_out(str(out / "saved-model"))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(out / "saved-model"))
    converter.optimizations = []
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_enable_resource_variables = True
    model_bytes = converter.convert()
    (out / "model.tflite").write_bytes(model_bytes)
    fb = schema.Model.GetRootAsModel(model_bytes, 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    operators = sorted({names[fb.OperatorCodes(i).BuiltinCode()] for i in range(fb.OperatorCodesLength())})
    assert all(not n.startswith("CUSTOM") for n in operators)
    infer = tf.function(lambda x: stream(x, training=False),
                        input_signature=[tf.TensorSpec([1, 3, 40], tf.float32)])
    rows, arrays = [], {}
    for condition, name in enumerate(frozen["synthetic_streams"]):
        inputs = synthetic(np, condition)
        for state in states:
            state.assign(tf.zeros_like(state))
        reference = np.asarray([float(infer(x).numpy()[0, 0]) for x in inputs], np.float32)
        repeats = []
        for _ in range(2):
            interpreter = Interpreter(model_content=model_bytes, num_threads=1,
                                      experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
            interpreter.allocate_tensors()
            inp, output = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
            assert inp["dtype"] == output["dtype"] == np.float32
            scores = []
            for x in inputs:
                interpreter.set_tensor(inp["index"], x)
                interpreter.invoke()
                scores.append(float(interpreter.get_tensor(output["index"])[0, 0]))
            repeats.append(np.asarray(scores, np.float32))
        error = float(np.max(np.abs(reference - repeats[0])))
        assert np.isfinite(reference).all() and np.isfinite(repeats).all()
        assert np.array_equal(*repeats) and error <= TOLERANCE
        arrays[name + "_input"] = inputs
        arrays[name + "_tf"] = reference
        arrays[name + "_lite"] = repeats[0]
        arrays[name + "_repeat"] = repeats[1]
        rows.append(dict(stream=name, steps=STEPS, input_sha256=sha(inputs.tobytes()),
                         tf_vs_lite_max_abs=error, two_fresh_repeats_exact=True))
    np.savez_compressed(out / "synthetic-arrays.npz", **arrays)
    assert all(np.array_equal(v.numpy(), before) for v, before in zip(original.variables, original_snapshot, strict=True))
    project = out / "project"
    (project / "main").mkdir(parents=True)
    (project / "components").mkdir()
    for component in frozen["identity"]["components"]:
        shutil.copytree(repo / "managed_components" / ("espressif__" + component),
                        project / "components" / component)
    cmake = project / "components/esp-tflite-micro/CMakeLists.txt"
    before = cmake.read_text()
    assert before.count("set(priv_req)") == 1
    cmake.write_text(before.replace("set(priv_req)", "set(pub_req esp-nn)\n\nset(priv_req)"))
    (project / "CMakeLists.txt").write_text('cmake_minimum_required(VERSION 3.16)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\nproject(nyx_fp32_feasibility)\n')
    (project / "main/CMakeLists.txt").write_text('idf_component_register(SRCS "probe.cpp" INCLUDE_DIRS "." REQUIRES esp-tflite-micro esp-nn esp_timer EMBED_FILES "model.tflite")\n')
    shutil.copyfile(out / "model.tflite", project / "main/model.tflite")
    shutil.copyfile(Path(__file__).with_name("fp32_runtime_feasibility_p4_v1.cpp"), project / "main/probe.cpp")
    defaults = repo / "research/wakeword_p4/sdkconfig.defaults"
    shutil.copyfile(defaults, project / "sdkconfig.defaults")
    header = "#pragma once\nstatic const float expected[4][256] = {\n"
    for name in frozen["synthetic_streams"]:
        header += "{" + ",".join(float(v).hex() + "f" for v in arrays[name + "_lite"]) + "},\n"
    (project / "main/expected.h").write_text(header + "};\n")
    report = dict(schema="mac-fp32-runtime-feasibility-export-v1", plan_sha256=sha(json.dumps(frozen, indent=2).encode() + b"\n"),
                  identity=frozen["identity"], random_weights_sha256=sha(b"".join(v.tobytes() for v in original_snapshot)),
                  parameters=original.count_params(), states=[dict(shape=list(v.shape), bytes=v.numpy().nbytes) for v in states],
                  model_bytes=len(model_bytes), model_sha256=sha(model_bytes), operators=operators, cases=rows,
                  arrays_sha256=sha((out / "synthetic-arrays.npz").read_bytes()), original_variables_unchanged=True,
                  p4_executed=False, candidate_qualified=False, limits=frozen["limits"])
    (out / "export-result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["parameters", "model_bytes", "operators", "cases"]}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "export"])
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    current = plan(a.repository_root.resolve(), a.upstream.resolve())
    current["identity"]["p4_probe_sha256"] = sha(Path(__file__).with_name("fp32_runtime_feasibility_p4_v1.cpp").read_bytes())
    current["identity"]["sdkconfig_defaults_sha256"] = sha((a.repository_root / "research/wakeword_p4/sdkconfig.defaults").read_bytes())
    if a.action == "plan":
        with a.plan.open("x") as f:
            f.write(json.dumps(current, indent=2) + "\n")
        print(sha(a.plan.read_bytes()))
    else:
        frozen = json.loads(a.plan.read_text())
        assert frozen == current, "frozen protocol or bound sources changed"
        a.output_dir.mkdir(exist_ok=False)
        export(a.repository_root.resolve(), a.upstream.resolve(), a.output_dir.resolve(), frozen)


if __name__ == "__main__":
    main()
