"""Freeze and execute real installed TFLM for all four original trained FP32s."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import traceback

import numpy as np

from intake_fp32_internal_trained_mac_v1 import H5, INPUT_SHA
from run_fp32_real_tflm_host_mac_v1 import FLAGS, TIMER_HEADER

PREFIX = "research/wakeword_p4/"
HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def plan(repo, packet, intake_path, prior):
    original = json.loads((repo / (PREFIX + "results/mac-fp32-real-tflm-host-plan-v1.json")).read_text())
    component = repo / "managed_components/espressif__esp-tflite-micro"
    tree = {str(p.relative_to(component)): sha(p.read_bytes()) for p in component.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(component).parts}
    assert len(tree) == original["component_tree"]["files"]
    assert sha(json.dumps(tree, sort_keys=True).encode()) == original["component_tree"]["sha256"]
    sources = dict(original["source_sha256"])
    extra = "tensorflow/lite/micro/kernels/expand_dims.cc"
    compile_db = json.loads((prior / "build/compile_commands.json").read_text())
    assert sum(row["file"].endswith("/" + extra) for row in compile_db) == 1
    assert extra not in sources
    sources[extra] = sha((component / extra).read_bytes())
    assert all(sha((component / k).read_bytes()) == v for k, v in sources.items())
    intake = json.loads(intake_path.read_text())
    assert intake["h5_bindings"] == H5 and intake["payloads_verified"] == 18
    files = {str(p.relative_to(packet)): sha(p.read_bytes()) for p in packet.rglob("*") if p.is_file()}
    manifest = json.loads((packet / "manifest.json").read_text())
    assert set(files) == set(manifest["files"]) | {"manifest.json"}
    assert files["manifest.json"] == intake["manifest_sha256"]
    assert all(files[k] == v["sha256"] for k, v in manifest["files"].items())
    assert files["inputs.bin"] == INPUT_SHA
    compiler = shutil.which("clang++")
    helper_names = ["run_fp32_real_tflm_host_mac_v1.py", "intake_fp32_internal_trained_mac_v1.py",
                    "inspect_fp32_internal_model_mac_v2.py", Path(__file__).name,
                    "fp32_trained_real_tflm_host_v1.cpp"]
    return dict(schema="mac-fp32-trained-real-tflm-host-plan-v1", component_tree=original["component_tree"],
                original_host_plan_sha256=sha((repo / (PREFIX + "results/mac-fp32-real-tflm-host-plan-v1.json")).read_bytes()),
                source_sha256=dict(sorted(sources.items())), packet_files=files,
                intake_sha256=sha(intake_path.read_bytes()), h5_bindings=H5,
                helper_sha256={n: sha((HERE / n).read_bytes()) for n in helper_names},
                timer_header_sha256=sha(TIMER_HEADER.encode()), compiler=compiler,
                compiler_version=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
                flags=FLAGS, platform=platform.platform(), models=list(H5),
                streams=["zeros", "constant26", "ramp257", "xorshift32"], steps_per_stream=256,
                fresh_interpreter_repeats=2, float_max_abs_ceiling=1e-4,
                registered_operators=12, expected_invocations=8192,
                limits="Original four trained raw sigmoid models, untouched GPU references/input bytes. Actual installed MicroInterpreter with twelve registered operators including real ExpandDims; original float sources ESP_NN=0 and chrono timer adapter. Full startup retained, two fresh zero-value-initialized arenas/stream. Host64-bit main/variable allocation is not P4 usage or timing, poisoned-memory evidence, fullstream/phase qualification, recognition or FAR. No audio/development/acceptance reads or retries/config selection.")


def run(repo, packet, frozen, frozen_path, output):
    component = repo / "managed_components/espressif__esp-tflite-micro"
    output.mkdir(exist_ok=False)
    try:
        (output / "esp_timer.h").write_text(TIMER_HEADER)
        executable = output / "fp32-trained-real-tflm"
        command = [frozen["compiler"], *frozen["flags"]]
        command += ["-I" + str(p) for p in [output, component, component / "third_party/gemmlowp",
                    component / "third_party/flatbuffers/include", component / "third_party/ruy", component / "third_party/kissfft"]]
        command += [str(HERE / "fp32_trained_real_tflm_host_v1.cpp")]
        command += [str(component / name) for name in frozen["source_sha256"]]
        command += ["-o", str(executable)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        write(output / "compile.json", vars(compiled))
        compiled.check_returncode()
        rows = []
        for name in frozen["models"]:
            folder = output / name
            folder.mkdir()
            proc = subprocess.run([str(executable), str(packet / name / "model.tflite"),
                                   str(packet / "inputs.bin"), str(folder / "outputs.bin")], capture_output=True, text=True)
            write(folder / "execution.json", vars(proc))
            print(name, proc.stdout, flush=True)
            proc.check_returncode()
            assert proc.stdout.count("HOST_TFLM_DONE") == 1
            cases = [json.loads(line.split(" ", 1)[1]) for line in proc.stdout.splitlines() if line.startswith("HOST_TFLM_CASE ")]
            assert [(c["condition"], c["repeat"], c["steps"]) for c in cases] == [(i, j, 256) for i in range(4) for j in range(2)]
            values = np.fromfile(folder / "outputs.bin", dtype="<f4").reshape(4, 2, 256)
            with np.load(packet / name / "references.npz", allow_pickle=False) as z:
                tf, lite = z["tf"], z["litert"]
            np.savez_compressed(folder / "saved-arrays.npz", tflm=values, tf=tf, litert=lite)
            finite = bool(np.isfinite(values).all())
            exact = values[:, 0].tobytes() == values[:, 1].tobytes()
            # Compare every step/repeat, including startup, using float64 differences.
            tf_error = float(np.max(np.abs(values.astype(np.float64) - tf.astype(np.float64))))
            lite_error = float(np.max(np.abs(values.astype(np.float64) - lite.astype(np.float64))))
            row = dict(model=name, model_sha256=frozen["packet_files"][name + "/model.tflite"],
                       original_h5_sha256=H5[name], original_references_sha256=frozen["packet_files"][name + "/references.npz"],
                       actual_registered_tflm_steps=2048, fresh_repeats_byte_exact=exact, finite=finite,
                       tflm_vs_tf_max_abs=tf_error, tflm_vs_litert_max_abs=lite_error, cases=cases,
                       outputs_sha256=sha((folder / "outputs.bin").read_bytes()),
                       saved_arrays_sha256=sha((folder / "saved-arrays.npz").read_bytes()),
                       execution_receipt_sha256=sha((folder / "execution.json").read_bytes()),
                       passed=finite and exact and max(tf_error, lite_error) <= frozen["float_max_abs_ceiling"])
            write(folder / "report.json", row)
            rows.append(row)
            assert row["passed"], name
        report = dict(schema="mac-fp32-trained-real-tflm-host-result-v1", plan_sha256=sha(frozen_path.read_bytes()),
                      actual_registered_tflm_steps=sum(r["actual_registered_tflm_steps"] for r in rows), models=rows,
                      executable_sha256=sha(executable.read_bytes()), compile_receipt_sha256=sha((output / "compile.json").read_bytes()),
                      all_passed=all(r["passed"] for r in rows), limits=frozen["limits"])
        write(output / "report.json", report)
        print(json.dumps(report), flush=True)
    except BaseException:
        (output / "failure.log").write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--intake-report", type=Path, required=True)
    p.add_argument("--prior-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    repo, packet = a.repository_root.resolve(), a.packet_root.resolve()
    current = plan(repo, packet, a.intake_report, a.prior_root)
    if a.action == "plan":
        write(a.plan, current)
        print(json.dumps(dict(plan_sha256=sha(a.plan.read_bytes()), translation_units=len(current["source_sha256"]))))
    else:
        frozen = json.loads(a.plan.read_text())
        assert frozen == current, "Frozen inputs/runtime/source/protocol changed"
        assert subprocess.check_output(["git", "show", "origin/research/hey-nyx-4090:" + str(a.plan.resolve().relative_to(repo))], cwd=repo) == a.plan.read_bytes()
        run(repo, packet, frozen, a.plan, a.output_dir.resolve())


if __name__ == "__main__":
    main()
