"""Record the completed synthetic FP32 full link without executing firmware."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    repo, run = a.repository_root, a.run_root
    build, project = run / "build", run / "project"
    frozen_bytes = (repo / "research/wakeword_p4/results/mac-fp32-runtime-feasibility-plan-v1.json").read_bytes()
    frozen = json.loads(frozen_bytes)
    exported = json.loads((run / "export-result.json").read_text())
    assert exported["plan_sha256"] == sha(frozen_bytes)
    assert exported["identity"] == frozen["identity"]
    assert sha((repo / "research/wakeword_p4/experiments/probe_fp32_runtime_feasibility_mac_v1.py").read_bytes()) == frozen["identity"]["script_sha256"]
    assert sha((project / "main/probe.cpp").read_bytes()) == frozen["identity"]["p4_probe_sha256"]
    assert sha((project / "sdkconfig.defaults").read_bytes()) == frozen["identity"]["sdkconfig_defaults_sha256"]
    for name, expected in frozen["identity"]["components"].items():
        source = repo / "managed_components" / ("espressif__" + name)
        copied = project / "components" / name
        original_files = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts}
        copied_files = {str(p.relative_to(copied)): p.read_bytes() for p in copied.rglob("*") if p.is_file() and ".git" not in p.relative_to(copied).parts}
        if name == "esp-tflite-micro":
            before = original_files["CMakeLists.txt"]
            assert copied_files["CMakeLists.txt"] == before.replace(b"set(priv_req)", b"set(pub_req esp-nn)\n\nset(priv_req)")
            copied_files["CMakeLists.txt"] = before
        assert copied_files == original_files
        assert len(original_files) == expected["files"]
        assert sha(json.dumps({k: sha(v) for k, v in original_files.items()}, sort_keys=True).encode()) == expected["sha256"]
    description = json.loads((build / "project_description.json").read_text())
    assert description["target"] == "esp32p4"
    commands_bytes = (build / "compile_commands.json").read_bytes()
    commands = json.loads(commands_bytes)
    main, = [row for row in commands if row["file"].endswith("/main/probe.cpp")]
    main_args = shlex.split(main["command"])
    compiler = Path(main_args[0])
    idf_version, = [arg.split("=", 1)[1].strip('"') for arg in main_args if arg.startswith("-DIDF_VER=")]
    assert idf_version == "v5.5.4"
    toolchain = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]
    flags = (build / "toolchain/cxxflags").read_text().split()
    assert "-mabi=ilp32f" in flags and any(f.startswith("-march=rv32imafc") for f in flags)
    elf = (build / "nyx_fp32_feasibility.elf").read_bytes()
    image = (build / "nyx_fp32_feasibility.bin").read_bytes()
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243
    model = (run / "model.tflite").read_bytes()
    assert sha(model) == exported["model_sha256"] and model in image
    log = run.with_name(run.name + "-build-v2.log").read_bytes()
    assert b"Successfully created esp32p4 image." in log and b"Smallest app partition is 0x100000 bytes" in log
    size = subprocess.check_output([str(compiler.with_name("riscv32-esp-elf-size")), "-A", str(build / "nyx_fp32_feasibility.elf")], text=True)
    (run / "elf-section-sizes.txt").write_text(size)
    sections = {parts[0]: int(parts[1]) for line in size.splitlines() if len(parts := line.split()) == 3 and parts[0].startswith(".")}
    report = dict(schema="mac-fp32-runtime-feasibility-p4-link-v1", script_sha256=sha(Path(__file__).read_bytes()),
                  prospective_plan_sha256=sha(frozen_bytes), export_report_sha256=sha((run / "export-result.json").read_bytes()),
                  original_export_session=60054, original_export_exit_code=0,
                  original_full_build_session=79066, original_full_build_exit_code=0,
                  initial_cli_failure=dict(session=9143, exit_code=2, reason="idf.py5.5.4 rejects -j before configuration or compilation", log_sha256=sha(run.with_name(run.name + "-build.log").read_bytes())),
                  idf=idf_version, toolchain=toolchain, architecture_flags=flags,
                  compile_commands_sha256=sha(commands_bytes), all_copied_runtime_sources_exact=True,
                  cmake_only_dependency_addition="set(pub_req esp-nn)",
                  elf_bytes=len(elf), elf_sha256=sha(elf), image_bytes=len(image), image_sha256=sha(image),
                  embedded_model_byte_exact=True, model_bytes=len(model),
                  static_dram_data_bss_bytes=sections[".dram0.data"] + sections[".dram0.bss"],
                  elf_sections=sections, build_log_sha256=sha(log),
                  actual_p4_execution=False, actual_arena_or_latency_measured=False,
                  candidate_qualified=False,
                  limits="Complete isolated P4 ELF/image link of random-weight FP32 context225 model and synthetic probe only. Static data/bss excludes dynamic arenas, stacks, allocations and panel workload. Standalone image size includes probe/runtime and cannot establish incremental panel flash headroom. No board access, audio, original model/data, recognition/FAR or changes to integer gates.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["idf", "toolchain", "image_bytes", "model_bytes", "static_dram_data_bss_bytes", "embedded_model_byte_exact", "actual_p4_execution"]}))


if __name__ == "__main__":
    main()
