"""Freeze and execute all original540 complete feature streams in real TFLM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = "research/wakeword_p4/"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    with path.open("x") as f:
        f.write(json.dumps(value, indent=2) + "\n")


def plan(repo, packet, intake_path, build):
    intake = json.loads(intake_path.read_text())
    assert intake["all_passed"] and len(intake["cases"]) == 540
    assert intake["unique_input_streams"] == 135 and intake["original_reference_steps"] == 96168
    files = {str(p.relative_to(packet)): sha(p.read_bytes()) for p in packet.rglob("*") if p.is_file()}
    manifest = json.loads((packet / "manifest.json").read_text())
    assert set(files) == set(manifest["files"]) | {"manifest.json"}
    assert files["manifest.json"] == intake["manifest_sha256"]
    assert all(files[n] == row["sha256"] for n, row in manifest["files"].items())
    build_report_path = repo / (PREFIX + "results/mac-fp32-internal-fullstream-host-build-v1.json")
    build_report = json.loads(build_report_path.read_text())
    assert sha((build / "fp32-internal-fullstream").read_bytes()) == build_report["executable_sha256"]
    assert sha((build / "compile.json").read_bytes()) == build_report["compile_receipt_sha256"]
    assert sha((build / "build-plan.json").read_bytes()) == build_report["build_plan_sha256"]
    native_plan = json.loads((build / "build-plan.json").read_text())
    assert native_plan["probe_sha256"] == sha((HERE / "fp32_internal_fullstream_host_v1.cpp").read_bytes())
    component = repo / "managed_components/espressif__esp-tflite-micro"
    tree = {str(p.relative_to(component)): sha(p.read_bytes()) for p in component.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(component).parts}
    assert len(tree) == build_report["component_tree"]["files"]
    assert sha(json.dumps(tree, sort_keys=True).encode()) == build_report["component_tree"]["sha256"]
    helpers = [Path(__file__).name, "fp32_internal_fullstream_host_v1.cpp", "build_fp32_internal_fullstream_host_mac_v1.py",
               "intake_fp32_internal_fullstream_mac_v1.py", "inspect_fp32_internal_model_mac_v2.py",
               "intake_fp32_internal_trained_mac_v1.py", "run_fp32_real_tflm_host_mac_v1.py"]
    return dict(schema="mac-fp32-internal-fullstream-host-execution-v1",
                intake_sha256=sha(intake_path.read_bytes()), packet_files=files,
                build_report_sha256=sha(build_report_path.read_bytes()), native_build=build_report,
                helper_sha256={n: sha((HERE / n).read_bytes()) for n in helpers}, cases=intake["cases"],
                expected_native_invocations=intake["expected_native_invocations"], fresh_repeats=2,
                max_abs=1e-4, phase_comparison_stream_offset=74, retries=0,
                limits="Original four unchanged raw FP32 models, original shared135 consumed numerical witnesses, all540 saved complete inputs and original TF/LiteRT/phase references. Actual installed TFLM compiled from57 translation units/12ops; two fresh value-initialized arenas per whole stream, no intermediate reset/crop/reorder/startup loss. All cross-runtime steps and both repeats retained; phase224+onezero additionally compared to stream[74:]. Nominal225 descriptive only. Host64-bit, no P4 memory/timing, recognition/FAR, fresh development/acceptance or promotion. No PCM decoding, playback or recording.")


def run(packet, build, frozen, plan_path, output):
    output.mkdir(exist_ok=False)
    try:
        executable = output / "fp32-internal-fullstream"
        shutil.copyfile(build / "fp32-internal-fullstream", executable)
        executable.chmod(0o755)
        assert sha(executable.read_bytes()) == frozen["native_build"]["executable_sha256"]
        models = output / "models"
        models.mkdir()
        for name in sorted(n for n in frozen["packet_files"] if n.startswith("models/")):
            data = (packet / name).read_bytes()
            assert sha(data) == frozen["packet_files"][name]
            (models / Path(name).name).write_bytes(data)
        rows, invocations = [], 0
        for case in frozen["cases"]:
            name, index, steps = case["model"], case["index"], case["steps"]
            folder = output / (name + f"-{index:03}")
            folder.mkdir()
            data = (packet / case["member"]).read_bytes()
            assert sha(data) == case["original_npz_sha256"]
            with np.load(packet / case["member"], allow_pickle=False) as z:
                raw_input = z["input"].astype("<f4").tobytes()
                reference_tf = np.stack([z["tf"], z["tf_repeat"]]).astype(np.float64)
                reference_lite = np.stack([z["litert"], z["litert_repeat"]]).astype(np.float64)
                phase = z["offline_phase"].astype(np.float64)
                nominal = z["offline_nominal"].astype(np.float64)
            assert len(raw_input) == steps * 480 and sha(raw_input) == case["input_sha256"]
            (folder / "input.bin").write_bytes(raw_input)
            proc = subprocess.run([str(executable), str(models / (name + ".tflite")),
                                   str(folder / "input.bin"), str(folder / "outputs.bin")], capture_output=True, text=True)
            write(folder / "execution.json", vars(proc))
            proc.check_returncode()
            lines = proc.stdout.splitlines()
            assert lines.count("HOST_TFLM_STREAM_DONE") == 1
            streams = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("HOST_TFLM_STREAM ")]
            assert [(r["repeat"], r["steps"]) for r in streams] == [(0, steps), (1, steps)]
            original_output = (folder / "outputs.bin").read_bytes()
            values = np.frombuffer(original_output, dtype="<f4").reshape(2, steps)
            finite = bool(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)))
            exact = values[0].tobytes() == values[1].tobytes()
            tf_gap = float(np.max(np.abs(values.astype(np.float64) - reference_tf)))
            lite_gap = float(np.max(np.abs(values.astype(np.float64) - reference_lite)))
            phase_gap = float(np.max(np.abs(values[:, 74:].astype(np.float64) - phase)))
            nominal_gap = float(np.max(np.abs(values[:, 74:].astype(np.float64) - nominal)))
            row = dict(model=name, index=index, request_order=case["request_order"], steps_per_repeat=steps,
                       actual_native_invocations=steps * 2, finite_probability=finite, fresh_repeats_byte_exact=exact,
                       tflm_vs_tf_max_abs=tf_gap, tflm_vs_litert_max_abs=lite_gap,
                       tflm_vs_phase_max_abs=phase_gap, nominal225_difference_descriptive=nominal_gap,
                       input_sha256=sha(raw_input), outputs_sha256=sha(original_output),
                       execution_receipt_sha256=sha((folder / "execution.json").read_bytes()), streams=streams,
                       passed=finite and exact and max(tf_gap, lite_gap, phase_gap) <= frozen["max_abs"])
            write(folder / "report.json", row)
            with (output / "cases.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            invocations += steps * 2
            if len(rows) % 25 == 0 or not row["passed"]:
                print(json.dumps(dict(cases_completed=len(rows), actual_native_invocations=invocations,
                                      numerical_failures=sum(not r["passed"] for r in rows))), flush=True)
        assert len(rows) == 540 and invocations == frozen["expected_native_invocations"] == 192336
        result = dict(schema="mac-fp32-internal-fullstream-host-result-v1", plan_sha256=sha(plan_path.read_bytes()),
                      executable_sha256=sha(executable.read_bytes()), cases=rows, actual_native_invocations=invocations,
                      all_passed=all(r["passed"] for r in rows), limits=frozen["limits"])
        write(output / "report.json", result)
        print(json.dumps(dict(cases=len(rows), actual_native_invocations=invocations, all_passed=result["all_passed"],
                              tf_max_abs=max(r["tflm_vs_tf_max_abs"] for r in rows),
                              litert_max_abs=max(r["tflm_vs_litert_max_abs"] for r in rows),
                              phase_max_abs=max(r["tflm_vs_phase_max_abs"] for r in rows))), flush=True)
        assert result["all_passed"], "Original fullstream numerical failures retained; no retry"
    except BaseException:
        (output / "failure.log").write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--intake-report", type=Path, required=True)
    p.add_argument("--build-root", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    a = p.parse_args()
    repo, packet, build = a.repository_root.resolve(), a.packet_root.resolve(), a.build_root.resolve()
    current = plan(repo, packet, a.intake_report, build)
    if a.action == "plan":
        write(a.plan, current)
        print(json.dumps(dict(plan_sha256=sha(a.plan.read_bytes()), cases=540, expected_native_invocations=current["expected_native_invocations"])))
    else:
        frozen = json.loads(a.plan.read_text())
        assert frozen == current, "Frozen runtime/source/input/reference identity changed"
        assert subprocess.check_output(["git", "show", "origin/research/hey-nyx-4090:" + str(a.plan.resolve().relative_to(repo))], cwd=repo) == a.plan.read_bytes()
        run(packet, build, frozen, a.plan, a.output_dir.resolve())


if __name__ == "__main__":
    main()
