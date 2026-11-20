"""Build the official Willow reference for the 10.1-inch rev1 board and check its OTA payload."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory
import zlib

from build_rev1_dev import (API, LOGIN, MAC_RECORD, NODE, PATCH, PROFILE, cache_values, check_sources,
                            elf_id, from_cache, from_image, require)
from check_builds import BOARD_COMMIT, MODEL, app_slots

WILLOW = PATCH.with_name("hey-willow-reference.patch")
NAME = "wn9_heywillow_tts"
MEMBERS = {
    "_MODEL_INFO_": "defc60162df4583975f7033a3994206574a8f8376ea2d2d9efc42d92af8ffca6",
    "wn9_data": "f3a661a2ba7d57ab96af8be01f1f3469e2da30bc1693b78fd5efd3114ea25977",
    "wn9_index": "c5557ca5d1ec6943540a6abbb3ea235a693f4637369f92e873ff566b73a61cfd",
}


def digest(data):
    return sha256(data).hexdigest()


def check_container(container, source):
    raw = zlib.decompress(container[4:])
    assert len(raw) == struct.unpack_from("<I", container)[0]
    count, name, members = struct.unpack_from("<I32sI", raw)
    assert (count, name.rstrip(b"\0").decode(), members) == (1, NAME, 3)
    offset = 40 + members * 40
    records = {}
    for i in range(members):
        name, start, size = struct.unpack_from("<32sII", raw, 40 + 40*i)
        name = name.rstrip(b"\0").decode()
        data = raw[start:start+size]
        assert start == offset and len(data) == size and data == (source / name).read_bytes()
        assert digest(data) == MEMBERS[name] and name not in records
        records[name] = dict(bytes=size, sha256=digest(data))
        offset += size
    assert offset == len(raw) and records.keys() == MEMBERS.keys()
    return records


def reference_image(board, build, services, engine):
    desc = json.loads((build / "project_description.json").read_text())
    settings = cache_values(build / "CMakeCache.txt")
    assert Path(desc["project_path"]).resolve() == board and desc["target"] == "esp32p4"
    assert desc["git_revision"] == "v5.5.4" and (int(desc["min_rev"]), int(desc["max_rev"])) == (100, 199)
    assert settings["PANEL_BOARD_PROFILE"] == PROFILE and settings["NYX_PANEL_WAKE_ENGINE"] == engine
    assert settings["NYX_PANEL_RESEARCH_FIXED_MARKER"] == "OFF"
    app = (build / desc["app_bin"]).read_bytes()
    elf = build / desc["app_elf"]
    assert struct.unpack_from("<HxHH", app, 12) == (18, 100, 199)
    assert app[23] == 1 and sha256(app[:-32]).digest() == app[-32:]
    assert elf_id(app) == digest(elf.read_bytes())
    if services:
        require(services == from_cache(build / "CMakeCache.txt") == from_image(app),
                "Source cache, effective cache and linked service profile must agree")
    else:
        assert settings[API] == settings[LOGIN] == "" and settings[NODE] == "OFF"
        assert b"wss://nyx-api.chrono-ai.fun/api/v1/nodes/ws\0" in app and b"https://nyxid.ai/login/device\0" in app

    table = (build / "partition_table/partition-table.bin").read_bytes()
    slots = app_slots(table)
    for slot in slots.values():
        slot["free_bytes"] = slot["bytes"] - len(app)
        assert slot["free_bytes"] >= 0
    metadata = dict(board_commit=BOARD_COMMIT, board_profile=PROFILE, idf=desc["git_revision"],
                chip="esp32p4", min_revision=100, max_revision=199,
                services="development" if services else "public", service_fingerprint=services.fingerprint if services else None,
                app_bytes=len(app), app_sha256=digest(app), descriptor_elf_sha256=elf_id(app),
                ota_slots=slots, partition_table_sha256=digest(table), marker_patch_sha256=digest(PATCH.read_bytes()))
    return desc, app, elf, metadata


def verify(board, build, services):
    desc, app, elf, metadata = reference_image(board, build, services, "hey-willow")
    sr = board / "managed_components/espressif__esp-sr"
    models = sr / "model/wakenet_model"
    container = (build / "esp-idf/panel_services/panel-hey-willow.bin").read_bytes()
    assert app.count(container) == 1
    members = check_container(container, models / NAME)
    c161 = (board / "models/hey-nyx-c161/model.tflite").read_bytes()
    assert digest(c161) == MODEL and c161 not in app
    spec = importlib.util.spec_from_file_location("embed_wake_model", board / "tools/embed_wake_model.py")
    packer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packer)
    with TemporaryDirectory() as tmp:
        hi = Path(tmp) / "hi.bin"
        packer.pack(models / "wn9_hiesp", hi)
        assert hi.read_bytes() not in app and (models / "wn9_hiesp/wn9_data").read_bytes() not in app
    nm = Path(desc["c_compiler"]).with_name("riscv32-esp-elf-nm")
    symbols = {line.split()[-1] for line in subprocess.check_output([nm, "--defined-only", elf], text=True).splitlines() if line.split()}
    assert {"esp_sr_wakenet9_quantized", "_binary_panel_hey_willow_bin_start"} <= symbols
    assert symbols.isdisjoint({"esp_wn_handle_from_name", "_binary_panel_hi_esp_bin_start", "board_marker_step",
                               "_binary_hey_nyx_model_bin_start", "panel_wake_runtime_iface"})
    for msg in ("panel.wake.willow", "first_pcm epoch=", "peer_send epoch=", "event epoch=%u model_end="):
        assert msg.encode() in app
    commands = json.loads((build / "compile_commands.json").read_text())
    for name in ("panel_voice.c", "panel_update.c"):
        cmd, = [entry["command"] for entry in commands if Path(entry["file"]).name == name]
        assert "-DNYX_PANEL_RESEARCH_WILLOW=1" in cmd and "-DNYX_PANEL_UPDATE_DISABLED=1" in cmd
    return dict(**metadata, esp_sr_component=(sr / ".component_hash").read_text().strip(),
                board_wakenet_library_sha256=digest((sr / "lib/esp32p4_less_v3/libwakenet.a").read_bytes()),
                model=NAME, members=members, container_sha256=digest(container), container_bytes=len(container),
                container_start_in_image=app.index(container), container_copies=1, mode="DET_MODE_95, DEFAULT",
                prime_samples=81600, willow_patch_sha256=digest(WILLOW.read_bytes()))


def run_build(engine, patch, verifier, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("board", type=Path)
    parser.add_argument("build", type=Path)
    env = parser.add_mutually_exclusive_group(required=True)
    env.add_argument("--old-cache", type=Path, help="service cache from the Mac C161 primary rev1 build")
    env.add_argument("--primary-app", type=Path, help="built Mac primary C161 app with SHA256 starting dd8af8a0")
    env.add_argument("--public", action="store_true", help="standard public service profile")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    require(__debug__, "Use normal Python for linked-image checks")
    board, build = args.board.resolve(), args.build.resolve()
    services = from_cache(args.old_cache) if args.old_cache else None
    if args.primary_app:
        primary = args.primary_app.read_bytes()
        identity = json.loads(MAC_RECORD.read_text())["diagnostic_app"]
        require(digest(primary) == identity["sha256"] and elf_id(primary) == identity["descriptor_elf_sha256"],
                "Expected the Mac primary C161 image recorded in Issue 18")
        services = from_image(primary)
    if services:
        require(services.fingerprint == "6f3d3f8368" and services.node_dev,
                "Expected the development service profile of the Mac primary C161 comparison")
    check_sources(board, extra_patches=(patch,))
    if not args.verify_only:
        build.mkdir(parents=True, exist_ok=True)
        require(not any(build.iterdir()), "Use an empty directory for a new build")
        service_settings = services.cmake() if services else {API: "", LOGIN: "", NODE: "OFF"}
        settings = dict(PANEL_BOARD_PROFILE=PROFILE, SDKCONFIG=str(build / "sdkconfig"),
                        NYX_PANEL_RESEARCH_FIXED_MARKER="OFF", NYX_PANEL_WAKE_ENGINE=engine, **service_settings)
        options = [word for k, v in settings.items() for word in ("-D", f"{k}={v}")]
        logfile = "willow-build.log" if engine == "hey-willow" else f"{engine}-build.log"
        with (build / logfile).open("w") as log:
            for command in (["idf.py", "-B", build, *options, "reconfigure"], ["cmake", "--build", build, "--parallel", "2"]):
                require(subprocess.run(command, cwd=board, stdout=log, stderr=subprocess.STDOUT).returncode == 0,
                        f"Inspect the local {logfile} for the build error")
    print(json.dumps(verifier(board, build, services), indent=2))


if __name__ == "__main__":
    run_build("hey-willow", WILLOW, verify, __doc__)
