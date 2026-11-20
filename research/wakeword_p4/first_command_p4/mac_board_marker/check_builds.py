"""Inspect the built app and generated partition table for each ESP32-P4 profile."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import struct
import subprocess

BOARD_COMMIT = "0d8e1935702a561b535a8c1a0921fd8034943d2c"
MODEL = "c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81"


def app_slots(table):
    slots = {}
    for offset in range(0, len(table), 32):
        magic, kind, subtype, start, size, label, _ = struct.unpack_from("<HBBII16sI", table, offset)
        if magic == 0x50aa and kind == 0:
            slots[label.rstrip(b"\0").decode()] = dict(offset=start, bytes=size)
    assert set(slots) == {"ota_0", "ota_1"}
    return slots


def image(build, gdb):
    desc = json.loads((build / "project_description.json").read_text())
    board = Path(desc["project_path"])
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=board, text=True).strip() == BOARD_COMMIT
    settings = (build / "CMakeCache.txt").read_text().splitlines()
    profile, = [line.split("=", 1)[1] for line in settings if line.startswith("PANEL_BOARD_PROFILE:STRING=")]
    assert "NYX_PANEL_RESEARCH_FIXED_MARKER:BOOL=ON" in settings
    assert "NYX_PANEL_WAKE_ENGINE:STRING=hey-nyx" in settings
    assert "NYX_PANEL_WAKE_RESEARCH_HANDOFF:PATH=" in settings
    assert desc["target"] == "esp32p4" and desc["git_revision"] == "v5.5.4"
    catalog = json.loads((board / "boards/catalog.json").read_text())
    expected = catalog["profiles"][profile]
    backend = catalog["boards"][expected["board"]]["backend"]
    assert (int(desc["min_rev"]), int(desc["max_rev"])) == (expected["min_revision"], expected["max_revision"])

    commands = json.loads((build / "compile_commands.json").read_text())
    for name in ("board_marker.cpp", "panel_voice.c", "panel_update.c"):
        cmd, = [entry["command"] for entry in commands if Path(entry["file"]).name == name]
        assert "-DNYX_PANEL_RESEARCH_FIXED_MARKER=1" in cmd and "-DNYX_PANEL_UPDATE_DISABLED=1" in cmd
    backend_path, = [Path(entry["file"]) for entry in commands if Path(entry["file"]).name == "product_backend.c"]
    assert backend_path == board / "boards" / backend / "product_backend.c"

    cfg = build / "esp-idf/panel_wake_runtime/generated"
    model = (board / "models/hey-nyx-c161/model.tflite").read_bytes()
    assert len(model) == 60952 and sha256(model).hexdigest() == MODEL
    assert (cfg / "hey_nyx_model.bin").read_bytes() == model
    defines = (cfg / "wake_config.h").read_text().splitlines()
    assert "#define PANEL_WAKE_THRESHOLD 0.91796875" in defines
    assert "#define PANEL_WAKE_SMOOTHING_STEPS 5" in defines
    assert "#define PANEL_WAKE_PRIME_SAMPLES 81440" in defines
    binary = (build / desc["app_bin"]).read_bytes()
    assert binary.count(model) == 1
    for message in (b"ref=physical tx=unavailable", b"decision epoch=", b"first_pcm epoch=", b"peer_send epoch="):
        assert message in binary
    elf = build / desc["app_elf"]
    nm = Path(desc["c_compiler"]).with_name("riscv32-esp-elf-nm")
    symbols = subprocess.check_output([nm, "--defined-only", elf], text=True)
    for name in ("board_marker_bytes", "board_marker_begin", "board_marker_step"):
        assert re.search(rf"^[0-9a-f]+ T {name}$", symbols, re.M)
    debug = subprocess.check_output([gdb, "-nx", "-batch", "-ex", "p sizeof(struct board_marker)", elf], text=True)
    state = int(re.search(r"\$1 = (\d+)", debug)[1])

    table = (build / "partition_table/partition-table.bin").read_bytes()
    slots = app_slots(table)
    smallest = min(row["bytes"] for row in slots.values())
    assert len(binary) <= smallest
    return dict(profile=profile, board=catalog["boards"][expected["board"]]["name"], backend=backend,
                min_rev=desc["min_rev"], max_rev=desc["max_rev"], app_bytes=len(binary),
                app_sha256=sha256(binary).hexdigest(), elf_sha256=sha256(elf.read_bytes()).hexdigest(),
                model_start_in_image=binary.index(model), model_copies=1, marker_state_bytes=state,
                partition_table_sha256=sha256(table).hexdigest(), app_slots=slots, slot_free_bytes=smallest-len(binary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", nargs="+", type=Path)
    parser.add_argument("--gdb", default="riscv32-esp-elf-gdb")
    args = parser.parse_args()
    rows = [image(path, args.gdb) for path in args.build]
    print(json.dumps(dict(board_commit=BOARD_COMMIT,
                          patch_sha256=sha256(Path(__file__).with_name("c161-fixed-marker.patch").read_bytes()).hexdigest(),
                          idf="v5.5.4", model_sha256=MODEL, model_bytes=60952,
                          threshold=0.91796875, smoothing_steps=5, records=rows), indent=2))


if __name__ == "__main__":
    main()
