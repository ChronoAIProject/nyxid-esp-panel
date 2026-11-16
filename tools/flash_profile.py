#!/usr/bin/env python3
"""Preserving app-only USB update with board, device and rollback checks.

Requires esptool 5.x. A chip revision identifies silicon, never board wiring:
the operator supplies the actual board profile and the intended P4 MAC.
"""
import argparse
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

from board_profiles import IMAGE_BOARD_OFFSET, catalog, image_identity, validate_image


def validate_target(image, profile, actual_mac, expected_mac, revision):
    p = validate_image(image, profile)
    if actual_mac.lower() != expected_mac.lower():
        raise ValueError("Connected P4 MAC does not match the intended device")
    if not p["min_revision"] <= revision <= p["max_revision"]:
        raise ValueError("Connected chip revision is outside the image profile")
    return p


def installed_identity(header):
    try:
        return image_identity(header)
    except ValueError:
        # An identified but malformed/unsupported image is not a first migration.
        # Preserve that distinction instead of hiding its failure as legacy.
        if header[IMAGE_BOARD_OFFSET:].startswith(b"NYX-BOARD-"):
            raise
        return None


def partition_layout(table):
    parts = {}
    for pos in range(0, 0xc00, 32):
        entry = table[pos:pos + 32]
        if len(entry) != 32 or entry[:2] != b"\xaa\x50":
            break
        _, kind, subtype, offset, size, label, _ = struct.unpack("<HBBII16sI", entry)
        name = label.rstrip(b"\0").decode("ascii")
        if name in parts:
            raise ValueError("Duplicate partition identity")
        parts[name] = {"offset": offset, "size": size, "kind": kind, "subtype": subtype}
    expected = {"nvs": (0x9000, 0xd2000, 1, 2), "otadata": (0xdc000, 0x2000, 1, 0),
                "ota_0": (0xe0000, 0x690000, 0, 16), "ota_1": (0x770000, 0x690000, 0, 17),
                "storage": (0xe00000, 0x200000, 1, 130)}
    for name, values in expected.items():
        if parts.get(name) != dict(zip(("offset", "size", "kind", "subtype"), values)):
            raise ValueError("Installed partition layout differs; explicit backed-up migration required")
    return parts


def active_slot(metadata):
    if len(metadata) != 8192:
        raise ValueError("Invalid OTA metadata size")
    entries = []
    for pos in (0, 4096):
        sequence, state, crc = struct.unpack_from("<I20xII", metadata, pos)
        if sequence not in (0, 0xffffffff) and state not in (3, 4) and crc == zlib.crc32(metadata[pos:pos + 4], 0xffffffff):
            entries.append((sequence, state))
    if not entries:
        if metadata != b"\xff" * 8192:
            raise ValueError("No valid OTA selection; refusing to guess active app")
        return "ota_0"
    sequence, state = max(entries)
    if state == 1:
        raise ValueError("Current OTA image is pending verification; wait for boot health check")
    return f"ota_{(sequence - 1) % 2}"


def read_bounded(esp, offset, size):
    # Some USB bridges corrupt the stub's 64-sector streaming read window.
    # Sector-sized requests use the stock checked protocol with one reply each.
    chunks = []
    for done in range(0, size, 4096):
        chunks.append(esp.read_flash(offset + done, min(4096, size - done)))
        if len(chunks) % 256 == 0:
            print(json.dumps({"stage": "reading_backup", "offset": offset,
                              "bytes": sum(map(len, chunks)), "total": size}), flush=True)
    return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--profile", choices=catalog()["profiles"], required=True)
    parser.add_argument("--expected-mac", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--baud", type=int, default=460800, help="Write baud rate")
    parser.add_argument("--read-baud", type=int, default=115200,
                        help="Conservative backup read baud rate for USB serial bridges")
    parser.add_argument("--rollback-image", type=Path,
                        help="Existing known image, verified against this device instead of reading the full app slot")
    args = parser.parse_args()
    image = args.image.read_bytes()
    validate_image(image, args.profile)
    if len(image) > 0x690000:
        parser.error("Application exceeds the OTA slot")
    # Import only when actually flashing; policy checks and --help need no SDK.
    try:
        import esptool
        from esptool.cmds import attach_flash, detect_flash_size, write_flash
    except ImportError as error:
        parser.error("Install tools/requirements-flash.txt in a separate virtual environment; "
                     "the ESP-IDF build environment's esptool 4.x API is incompatible. "
                     f"({error})")
    os.umask(0o077)
    root = args.backup_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    with esptool.detect_chip(args.port, baud=115200) as chip:
        mac = ":".join(f"{b:02x}" for b in chip.read_mac())
        if chip.CHIP_NAME != "ESP32-P4":
            raise ValueError("Unsupported chip")
        validate_target(image, args.profile, mac, args.expected_mac, chip.get_chip_revision())
        esp = chip.run_stub()
        esp.change_baud(args.read_baud)
        attach_flash(esp)
        if detect_flash_size(esp) not in ("16MB", "32MB"):
            raise ValueError("Insufficient or unrecognized flash capacity")
        table = read_bounded(esp, 0x8000, 4096)
        parts = partition_layout(table)
        ota = parts["otadata"]
        metadata = read_bounded(esp, ota["offset"], ota["size"])
        selected = active_slot(metadata)
        app = parts[selected]
        old_header = read_bounded(esp, app["offset"], 4096)
        # Before this feature, installed images had no board descriptor. The
        # operator's explicit profile/MAC identifies that first migration.
        current = installed_identity(old_header)
        if current and current["profile"] != args.profile:
            raise ValueError("Installed board profile differs; this is not a preserving update")
        # A supplied backup must match flash on this exact device before use.
        backup = args.rollback_image.read_bytes() if args.rollback_image else read_bounded(esp, app["offset"], app["size"])
        if len(backup) < 288 or len(backup) > app["size"] or backup[0] != 0xe9:
            raise ValueError("Invalid rollback image")
        if esp.flash_md5sum(app["offset"], len(backup)).lower() != hashlib.md5(backup).hexdigest():
            raise ValueError("Rollback image does not match the installed app")
        (root / "app-before-private.bin").write_bytes(backup)
        prefix = read_bounded(esp, 0x2000, 0xe0000 - 0x2000)
        (root / "prefix-nvs-before-private.bin").write_bytes(prefix)
        protected = [{"offset": 0x2000, "size": len(prefix)}, parts["storage"]]
        for region in protected:
            region["md5"] = esp.flash_md5sum(region["offset"], region["size"]).lower()
        record = {"profile": args.profile, "mac": mac, "slot": selected, "offset": app["offset"],
                  "previous_profile": current["profile"] if current else "legacy-unidentified",
                  "image_sha256": hashlib.sha256(image).hexdigest(), "protected": protected}
        (root / "plan.json").write_text(json.dumps(record, indent=2) + "\n")
        esp.change_baud(args.baud)
        write_flash(esp, [(app["offset"], str(args.image.resolve()))], flash_freq="keep",
                    flash_mode="keep", flash_size="keep", compress=True)
        if esp.flash_md5sum(app["offset"], len(image)).lower() != hashlib.md5(image).hexdigest():
            raise ValueError("Flashed app verification failed")
        for region in protected:
            if esp.flash_md5sum(region["offset"], region["size"]).lower() != region["md5"]:
                raise ValueError("Persistent region changed unexpectedly")
        record["app_and_persistent_regions_verified"] = True
        (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        esp.hard_reset()
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
