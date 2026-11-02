#!/usr/bin/env python3
"""Extract this device's runtime NVS before changing partition offsets.

Input is a complete, private 16 MiB esptool read_flash backup. Neither the
backup nor extracted settings should be placed in version control.
"""
import argparse
import hashlib
import json
import pathlib
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    data = args.backup.read_bytes()
    if len(data) != 16 * 1024 * 1024:
        parser.error("expected a complete 16 MiB backup")
    partition = None
    for position in range(0x8000, 0x8C00, 32):
        entry = data[position:position + 32]
        if entry[:2] != b"\xaa\x50":
            break
        _, kind, subtype, offset, size, name, _ = struct.unpack("<HBBII16sI", entry)
        if name.rstrip(b"\0") == b"nvs" and kind == 1 and subtype == 2:
            partition = offset, size
    if not partition:
        parser.error("runtime nvs partition not found")
    offset, size = partition
    if size > 0xD2000 or offset + size > len(data):
        parser.error("source NVS does not fit target partition; do not truncate it")
    settings = data[offset:offset + size] + b"\xff" * (0xD2000 - size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(settings)
    args.output.chmod(0o600)
    print(json.dumps({"source_offset": hex(offset), "source_size": size,
                      "target_offset": "0x9000", "target_size": len(settings),
                      "backup_sha256": hashlib.sha256(data).hexdigest(),
                      "nvs_sha256": hashlib.sha256(settings).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
