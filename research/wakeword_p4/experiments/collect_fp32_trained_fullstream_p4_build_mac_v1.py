"""Authenticate the completed trained-model P4 link; never contact hardware."""
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
    for name in ("repository-root", "run-root", "report"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    repo, run = a.repository_root.resolve(), a.run_root.resolve()
    plan_path = repo / "research/wakeword_p4/results/mac-fp32-trained-fullstream-p4-preparation-v1.json"
    plan_bytes = plan_path.read_bytes()
    assert plan_bytes == (run / "fixture-plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    assert sha((repo / "research/wakeword_p4/experiments/prepare_fp32_trained_fullstream_p4_mac_v1.py").read_bytes()) == plan["source_sha256"]
    assert sha((repo / "research/wakeword_p4/experiments/fp32_trained_fullstream_p4_v1.cpp").read_bytes()) == plan["probe_sha256"]
    for name, expected in plan["generated_files"].items():
        data = (run / name).read_bytes()
        assert len(data) == expected["bytes"] and sha(data) == expected["sha256"]
    for name, expected in plan["components"].items():
        original = repo / "managed_components" / ("espressif__" + name)
        copied = run / "project/components" / name
        before = {str(p.relative_to(original)): p.read_bytes() for p in original.rglob("*") if p.is_file() and ".git" not in p.relative_to(original).parts}
        after = {str(p.relative_to(copied)): p.read_bytes() for p in copied.rglob("*") if p.is_file() and ".git" not in p.relative_to(copied).parts}
        if name == "esp-tflite-micro":
            assert after["CMakeLists.txt"] == before["CMakeLists.txt"].replace(b"set(priv_req)", b"set(pub_req esp-nn)\n\nset(priv_req)")
            after["CMakeLists.txt"] = before["CMakeLists.txt"]
        assert before == after and len(before) == expected["files"]
        assert sha(json.dumps({k: sha(v) for k, v in before.items()}, sort_keys=True).encode()) == expected["sha256"]
    build = run / "build"
    description = json.loads((build / "project_description.json").read_bytes())
    assert description["target"] == "esp32p4"
    command_bytes = (build / "compile_commands.json").read_bytes()
    commands = json.loads(command_bytes)
    main_command, = [r for r in commands if r["file"] == str(run / "project/main/probe.cpp")]
    args = shlex.split(main_command["command"])
    compiler = Path(args[0])
    version = subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]
    assert "esp-14.2.0_20260121" in version
    idf, = [s.split("=", 1)[1].strip('"') for s in args if s.startswith("-DIDF_VER=")]
    assert idf == "v5.5.4"
    flags = (build / "toolchain/cxxflags").read_text().split()
    assert "-mabi=ilp32f" in flags and any(s.startswith("-march=rv32imafc") for s in flags)
    elf_path = build / "nyx_trained_fp32_probe.elf"
    image = (build / "nyx_trained_fp32_probe.bin").read_bytes()
    elf = elf_path.read_bytes()
    assert elf[:4] == b"\x7fELF" and int.from_bytes(elf[18:20], "little") == 243
    assert len(image) <= 0x400000
    for name, expected in plan["assets"].items():
        data = (run / "project/main" / name).read_bytes()
        assert data in image and sha(data) == expected["sha256"]
    log = (run / "build.log").read_bytes()
    assert b"Successfully created esp32p4 image." in log and b"Smallest app partition is 0x400000 bytes" in log
    sizes = subprocess.check_output([str(compiler.with_name("riscv32-esp-elf-size")), "-A", str(elf_path)], text=True)
    (run / "elf-section-sizes.txt").write_text(sizes)
    sections = {s[0]: int(s[1]) for line in sizes.splitlines() if len(s := line.split()) == 3 and s[0].startswith(".")}
    result = dict(schema="mac-fp32-trained-fullstream-p4-link-v1", source_sha256=sha(Path(__file__).read_bytes()),
                  preparation_sha256=sha(plan_bytes), original_build_session=7488, original_build_exit=0,
                  idf=idf, toolchain=version, architecture_flags=flags,
                  compile_commands_sha256=sha(command_bytes), runtime_components_exact=True,
                  elf_bytes=len(elf), elf_sha256=sha(elf), image_bytes=len(image), image_sha256=sha(image),
                  embedded_models=4, embedded_streams=6, all_embedded_assets_byte_exact=True,
                  prepared_executions=48, prepared_invocations=plan["expected_invocations"],
                  static_dram_data_bss_bytes=sections[".dram0.data"] + sections[".dram0.bss"],
                  elf_sections=sections, build_log_sha256=sha(log), hardware_accessed=False,
                  actual_p4_invocations=0, measured_arena_or_latency=False, candidate_qualified=False,
                  limits="Actual P4 cross-build/link and byte inclusion only. No board access, timing or arena measurements. Static DRAM excludes dynamic arenas, stacks and full-panel/audio workload. Future app-only use needs a current verified development board/slot/backup, fitting image and finally restoration; never flash generated partition/bootloader or dining-room production board.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("idf", "toolchain", "image_bytes", "embedded_models", "embedded_streams", "prepared_invocations", "static_dram_data_bss_bytes", "actual_p4_invocations")}))


if __name__ == "__main__":
    main()
