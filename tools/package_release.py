#!/usr/bin/env python3
"""Package a verified, production-only ESP-IDF build for GitHub Release OTA."""
import argparse
import hashlib
import json
import re
import shutil
import struct
import zipfile
from pathlib import Path

PROJECT = "nyxid-esp-panel"
ROOT = "https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download"
PROFILES = {"p4-rev1-16mb": 100, "p4-rev3-16mb": 300}
SLOT_SIZE = 0x690000


def package(build: Path, output: Path, profile: str, version: str):
    if not re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version):
        raise ValueError("Release must use a stable major.minor.patch version")
    if any(int(part) > 65535 for part in version.split(".")):
        raise ValueError("Version component too large")
    config = json.loads((build / "config/sdkconfig.json").read_text())
    if config.get("ESPTOOLPY_FLASHSIZE") != "16MB" or not config.get("BOOTLOADER_APP_ROLLBACK_ENABLE"):
        raise ValueError("Release requires 16 MB flash and boot rollback")
    if config.get("ESP32P4_REV_MIN_FULL") != PROFILES[profile]:
        raise ValueError("Build profile does not match minimum chip revision")
    # The generated compilation database records all CMake command-line defines.
    compile_commands = (build / "compile_commands.json").read_text()
    if any(marker in compile_commands for marker in (
        "NYX_AGENT_BASE_URL=", "HA_CAM_NYX_BASE_URL=", "NYX_PANEL_UPDATE_DISABLED=1"
    )):
        raise ValueError("Development firmware must never enter the stable channel")
    image = (build / f"{PROJECT}.bin").read_bytes()
    if len(image) > SLOT_SIZE or len(image) < 288 or image[0] != 0xE9:
        raise ValueError("Invalid or oversized application image")
    # IDF app descriptor follows the image header and first segment header.
    magic = struct.unpack_from("<I", image, 32)[0]
    actual_version = image[48:80].split(b"\0", 1)[0].decode("ascii")
    actual_project = image[80:112].split(b"\0", 1)[0].decode("ascii")
    if magic != 0xABCD5432 or actual_version != version or actual_project != PROJECT:
        raise ValueError("Application descriptor does not match release identity")
    output.mkdir(parents=True, exist_ok=True)
    name = f"{PROJECT}-{profile}.bin"
    shutil.copyfile(build / f"{PROJECT}.bin", output / name)
    manifest = dict(schema=1, project=PROJECT, profile=profile, version=version,
                    url=f"{ROOT}/v{version}/{name}", size=len(image),
                    sha256=hashlib.sha256(image).hexdigest())
    (output / f"update-{profile}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output / f"{PROJECT}-{profile}-factory.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for item in (f"{PROJECT}.bin", "bootloader/bootloader.bin", "partition_table/partition-table.bin",
                     "ota_data_initial.bin", "storage.bin", "flash_args"):
            archive.write(build / item, item)
    files = sorted(p for p in output.iterdir() if p.is_file() and profile in p.name and not p.name.startswith("checksums-"))
    (output / f"checksums-{profile}.txt").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--version-file", type=Path, default=Path("VERSION"))
    parser.add_argument("--tag")
    args = parser.parse_args()
    version = args.version_file.read_text().strip()
    if args.tag is not None and args.tag != f"v{version}":
        parser.error("Git tag must exactly match VERSION")
    print(json.dumps(package(args.build, args.output, args.profile, version), indent=2))


if __name__ == "__main__":
    main()
