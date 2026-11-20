"""Verify and package the public rev1 and LCD-X8 applications for the 400 ms playback trial."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import build_rev1_dev as dev
import check_builds

PROFILES = ("p4-rev1-16mb", "p4-rev3-lcd-x8-16mb")


def package(board, build_root, gdb, output):
    dev.require(__debug__, "Run with normal Python to enable image checks")
    dev.check_sources(board, extra_patches=dev.REFERENCE_BARGE)
    patches = []
    for path in (dev.PATCH, *dev.REFERENCE_BARGE):
        git = ["git", "-C", str(path.parent)]
        origin = subprocess.check_output(git + ["log", "-1", "--format=%H", "--", path.name], text=True).strip()
        content = subprocess.check_output(git + ["show", f"{origin}:./{path.name}"])
        assert content == path.read_bytes()
        patches.append(dict(file=path.name, source_commit=origin, sha256=sha256(content).hexdigest()))
    records, members = [], {}
    for profile in PROFILES:
        build = build_root / ("build-" + profile)
        desc = json.loads((build / "project_description.json").read_text())
        assert Path(desc["project_path"]).resolve() == board
        image = check_builds.image(build, gdb)
        app = (build / desc["app_bin"]).read_bytes()
        assert image["profile"] == profile and image["marker_state_bytes"] == 13304
        assert dev.elf_id(app) == image["elf_sha256"]
        assert app[23] == 1 and sha256(app[:-32]).digest() == app[-32:]
        assert struct.unpack_from("<HxHH", app, 12) == (18, int(image["min_rev"]), int(image["max_rev"]))
        cache = dev.cache_values(build / "CMakeCache.txt")
        assert cache[dev.API] == cache[dev.LOGIN] == "" and cache[dev.NODE] == "OFF"
        assert all(v + b"\0" in app for v in (b"https://nyx-api.chrono-ai.fun", b"https://nyxid.ai/login/agent-key",
                                             b"wss://nyx-api.chrono-ai.fun/api/v1/nodes/ws", b"https://nyxid.ai/login/device",
                                             b"nyx_agent", b"panel_voice"))
        assert "CONFIG_SPIRAM=y" in (build / "sdkconfig").read_text().splitlines()
        name = f"c161-reference-rearm-{profile}-{image['app_sha256'][:8]}.bin"
        members[name] = app
        records.append(dict(file=name, **image, descriptor_elf_sha256=dev.elf_id(app),
                            appended_sha256=app[-32:].hex(), psram_enabled=True))
    manifest = dict(board_commit=check_builds.BOARD_COMMIT, patches=patches, idf="v5.5.4",
                    service_profile="public", model_bytes=60952, model_sha256=check_builds.MODEL,
                    threshold=0.91796875, smoothing_steps=5, artifacts=records)
    members["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    temporary = output.with_suffix(".tmp")
    with ZipFile(temporary, "w") as dest:
        for name, body in members.items():
            dest.writestr(ZipInfo(name), body, compress_type=ZIP_DEFLATED)
    with ZipFile(temporary) as archive:
        assert archive.namelist() == list(members)
        for name, body in members.items():
            assert archive.read(name) == body
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("build_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gdb", default="riscv32-esp-elf-gdb")
    args = parser.parse_args()
    package(args.board.resolve(), args.build_root, args.gdb, args.output)
