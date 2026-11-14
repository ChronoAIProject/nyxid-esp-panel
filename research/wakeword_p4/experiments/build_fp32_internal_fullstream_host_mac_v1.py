"""Build actual fullstream MicroInterpreter probe; never access models or media."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from run_fp32_real_tflm_host_mac_v1 import TIMER_HEADER


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    repo = a.repository_root.resolve()
    component = repo / "managed_components/espressif__esp-tflite-micro"
    previous = repo / "research/wakeword_p4/results/mac-fp32-trained-real-tflm-host-plan-v1.json"
    frozen = json.loads(previous.read_text())
    tree = {str(p.relative_to(component)): sha(p.read_bytes()) for p in component.rglob("*")
            if p.is_file() and ".git" not in p.relative_to(component).parts}
    assert len(tree) == frozen["component_tree"]["files"]
    assert sha(json.dumps(tree, sort_keys=True).encode()) == frozen["component_tree"]["sha256"]
    assert len(frozen["source_sha256"]) == 57
    assert all(tree[k] == v for k, v in frozen["source_sha256"].items())
    assert subprocess.check_output([frozen["compiler"], "--version"], text=True).splitlines()[0] == frozen["compiler_version"]
    a.output_dir.mkdir(exist_ok=False)
    (a.output_dir / "esp_timer.h").write_text(TIMER_HEADER)
    source = Path(__file__).with_name("fp32_internal_fullstream_host_v1.cpp").resolve()
    executable = a.output_dir / "fp32-internal-fullstream"
    command = [frozen["compiler"], *frozen["flags"]]
    command += ["-I" + str(p) for p in [a.output_dir, component, component / "third_party/gemmlowp",
                component / "third_party/flatbuffers/include", component / "third_party/ruy", component / "third_party/kissfft"]]
    command += [str(source)] + [str(component / name) for name in frozen["source_sha256"]]
    command += ["-o", str(executable)]
    before = dict(schema="mac-fp32-internal-fullstream-host-build-plan-v1",
                  compiler=frozen["compiler"], compiler_version=frozen["compiler_version"], flags=frozen["flags"],
                  component_tree=frozen["component_tree"], source_sha256=frozen["source_sha256"],
                  probe_sha256=sha(source.read_bytes()), builder_sha256=sha(Path(__file__).read_bytes()),
                  timer_header_sha256=sha(TIMER_HEADER.encode()), command=command,
                  prior_plan_sha256=sha(previous.read_bytes()),
                  limits="Build-only plan, not a fullstream inference lock. Fixed concrete ABI/12 operators; one entire variable-length feature stream, two fresh value-initialized interpreter arenas. No input crop/reorder/intermediate reset. No models/references/audio/development/acceptance accessed. Actual inference must bind all540 original input/reference/model bytes separately before execution.")
    with (a.output_dir / "build-plan.json").open("x") as f:
        f.write(json.dumps(before, indent=2) + "\n")
    proc = subprocess.run(command, capture_output=True, text=True)
    with (a.output_dir / "compile.json").open("x") as f:
        f.write(json.dumps(vars(proc), indent=2) + "\n")
    proc.check_returncode()
    report = dict(schema="mac-fp32-internal-fullstream-host-build-v1", built=True,
                  build_plan_sha256=sha((a.output_dir / "build-plan.json").read_bytes()),
                  compile_receipt_sha256=sha((a.output_dir / "compile.json").read_bytes()),
                  executable_sha256=sha(executable.read_bytes()), probe_sha256=before["probe_sha256"],
                  builder_sha256=before["builder_sha256"], component_tree=frozen["component_tree"],
                  actual_translation_units=57, compile_exit=proc.returncode, model_invocations=0,
                  limits=before["limits"])
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
