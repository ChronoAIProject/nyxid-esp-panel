"""Predeclare and run real TFLM on the original candidate-free FP32 vectors."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess

PREFIX = "research/wakeword_p4/"
FLAGS = ["-std=c++17", "-O2", "-DNDEBUG", "-DTF_LITE_STATIC_MEMORY", "-DTF_LITE_DISABLE_X86_NEON", "-DESP_NN=0", "-ffp-contract=off", "-fno-rtti", "-fno-exceptions"]
TIMER_HEADER = """#pragma once
#include <chrono>
inline long long esp_timer_get_time() {
 return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_plan(repo, prior):
    original = json.loads((repo / (PREFIX + "results/mac-fp32-runtime-feasibility-export-v1.json")).read_text())
    assert sha((prior / "model.tflite").read_bytes()) == original["model_sha256"]
    assert sha((prior / "synthetic-arrays.npz").read_bytes()) == original["arrays_sha256"]
    component = repo / "managed_components/espressif__esp-tflite-micro"
    compile_db = json.loads((prior / "build/compile_commands.json").read_text())
    kernels = {"call_once.cc", "var_handle.cc", "read_variable.cc", "reshape.cc", "concatenation.cc",
               "strided_slice.cc", "assign_variable.cc", "logistic.cc", "kernel_util.cc",
               "conv_common.cc", "depthwise_conv_common.cc", "fully_connected_common.cc", "logistic_common.cc",
               "reshape_common.cc", "strided_slice_common.cc"}
    sources = {}
    for row in compile_db:
        needle = "/components/esp-tflite-micro/"
        if needle not in row["file"]:
            continue
        relative = row["file"].split(needle, 1)[1]
        path = Path(relative)
        selected = relative.startswith("tensorflow/lite/micro/") and (
            "/kernels/" not in relative or
            (path.parent.as_posix() == "tensorflow/lite/micro/kernels" and path.name in kernels) or
            relative in {"tensorflow/lite/micro/kernels/esp_nn/conv.cc", "tensorflow/lite/micro/kernels/esp_nn/depthwise_conv.cc", "tensorflow/lite/micro/kernels/esp_nn/fully_connected.cc"})
        selected |= relative.startswith(("tensorflow/lite/core/", "tensorflow/lite/kernels/", "tensorflow/compiler/mlir/lite/"))
        selected |= relative == "tensorflow/lite/c/common.cc"
        if selected:
            assert (component / relative).is_file()
            sources[relative] = sha((component / relative).read_bytes())
    assert len(sources) >= 50
    # All runtime headers/sources, including inline reference kernels, are bound.
    tree = {str(p.relative_to(component)): sha(p.read_bytes()) for p in component.rglob("*") if p.is_file() and ".git" not in p.relative_to(component).parts}
    expected_tree = original["identity"]["components"]["esp-tflite-micro"]
    assert len(tree) == expected_tree["files"] and sha(json.dumps(tree, sort_keys=True).encode()) == expected_tree["sha256"]
    compiler = shutil.which("clang++")
    return dict(schema="mac-fp32-real-tflm-host-plan-v1", original_model_sha256=original["model_sha256"],
                original_arrays_sha256=original["arrays_sha256"], original_export_report_sha256=sha((prior / "export-result.json").read_bytes()),
                component_tree=expected_tree, source_sha256=dict(sorted(sources.items())),
                script_sha256=sha(Path(__file__).read_bytes()), probe_sha256=sha(Path(__file__).with_name("fp32_real_tflm_host_v1.cpp").read_bytes()),
                timer_header_sha256=sha(TIMER_HEADER.encode()), flags=FLAGS,
                compiler=compiler, compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                platform=platform.platform(), streams=["zeros", "constant26", "ramp257", "xorshift32"], steps_per_stream=256,
                fresh_interpreter_repeats=2, float_max_abs_ceiling=1e-4,
                limits="Original seeded-random FP32 model and original synthetic input/reference arrays only. Actual installed MicroInterpreter/allocator/resource variables and eleven registered kernels; same esp_nn float source files compiled ESP_NN=0, excluding unexecuted integer acceleration; only esp_timer is a host chrono adapter. No fake interpreter, kernel math patch, candidate weights, audio/data scoring, board access or integer-gate change. Host64-bit allocation/outputs do not prove P4 allocation, memory or latency.")


def run(repo, prior, frozen, out):
    import numpy as np
    out.mkdir(exist_ok=False)
    component = repo / "managed_components/espressif__esp-tflite-micro"
    (out / "esp_timer.h").write_text(TIMER_HEADER)
    with np.load(prior / "synthetic-arrays.npz", allow_pickle=False) as z:
        inputs = np.concatenate([z[name + "_input"] for name in frozen["streams"]])
        assert inputs.dtype == np.float32 and inputs.shape == (1024, 1, 3, 40)
        inputs.astype("<f4").tofile(out / "inputs.bin")
        tf = np.stack([z[name + "_tf"] for name in frozen["streams"]])
        lite = np.stack([z[name + "_lite"] for name in frozen["streams"]])
    executable = out / "fp32-real-tflm"
    command = [frozen["compiler"], *FLAGS]
    command += ["-I" + str(p) for p in [out, component, component / "third_party/gemmlowp", component / "third_party/flatbuffers/include", component / "third_party/ruy", component / "third_party/kissfft"]]
    command += [str(Path(__file__).with_name("fp32_real_tflm_host_v1.cpp"))]
    command += [str(component / name) for name in frozen["source_sha256"]]
    command += ["-o", str(executable)]
    result = subprocess.run(command, capture_output=True, text=True)
    (out / "compile.json").write_text(json.dumps(vars(result), indent=2) + "\n")
    result.check_returncode()
    result = subprocess.run([str(executable), str(prior / "model.tflite"), str(out / "inputs.bin"), str(out / "outputs.bin")], capture_output=True, text=True)
    (out / "execution.json").write_text(json.dumps(vars(result), indent=2) + "\n")
    print(result.stdout, flush=True)
    result.check_returncode()
    assert result.stdout.count("HOST_TFLM_DONE") == 1
    cases = [json.loads(line.split(" ", 1)[1]) for line in result.stdout.splitlines() if line.startswith("HOST_TFLM_CASE ")]
    assert [(r["condition"], r["repeat"], r["steps"]) for r in cases] == [(i, j, 256) for i in range(4) for j in range(2)]
    values = np.fromfile(out / "outputs.bin", dtype="<f4").reshape(4, 2, 256)
    np.savez_compressed(out / "saved-arrays.npz", tflm=values, tf=tf, lite=lite)
    assert np.isfinite(values).all()
    exact = np.array_equal(values[:, 0], values[:, 1])
    lite_error = float(np.max(np.abs(values[:, 0] - lite)))
    tf_error = float(np.max(np.abs(values[:, 0] - tf)))
    assert exact and max(lite_error, tf_error) <= frozen["float_max_abs_ceiling"]
    report = dict(schema="mac-fp32-real-tflm-host-result-v1", plan_sha256=sha((json.dumps(frozen, indent=2) + "\n").encode()),
                  original_model_sha256=frozen["original_model_sha256"], original_arrays_sha256=frozen["original_arrays_sha256"],
                  actual_registered_tflm_steps=2048, fresh_repeats_exact=exact, tflm_vs_lite_max_abs=lite_error,
                  tflm_vs_tf_max_abs=tf_error, cases=cases, all_checks_pass=True,
                  inputs_sha256=sha((out / "inputs.bin").read_bytes()), outputs_sha256=sha((out / "outputs.bin").read_bytes()),
                  saved_arrays_sha256=sha((out / "saved-arrays.npz").read_bytes()),
                  executable_sha256=sha(executable.read_bytes()), compile_receipt_sha256=sha((out / "compile.json").read_bytes()),
                  execution_receipt_sha256=sha((out / "execution.json").read_bytes()), limits=frozen["limits"])
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--prior-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    current = make_plan(a.repository_root.resolve(), a.prior_root.resolve())
    if a.action == "plan":
        with a.plan.open("x") as f: f.write(json.dumps(current, indent=2) + "\n")
        print(json.dumps(dict(plan_sha256=sha(a.plan.read_bytes()), actual_tflm_translation_units=len(current["source_sha256"]))))
    else:
        frozen = json.loads(a.plan.read_text())
        assert frozen == current, "bound sources or protocol changed"
        run(a.repository_root.resolve(), a.prior_root.resolve(), frozen, a.output_dir.resolve())


if __name__ == "__main__":
    main()
