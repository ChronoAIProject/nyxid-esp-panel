#!/usr/bin/env python3
"""Pack the pinned ESP-SR Hi ESP model into the application OTA image."""
import argparse
from pathlib import Path
import struct
import zlib


def pack(source: Path, output: Path):
    files = sorted(p for p in source.iterdir() if p.is_file())
    if {p.name for p in files} != {"_MODEL_INFO_", "wn9_index", "wn9_data"}:
        raise ValueError("Unexpected Hi ESP model files")
    offset = 4 + 36 + 40 * len(files)
    header = struct.pack("<I32sI", 1, b"wn9_hiesp", len(files))
    payload = b""
    for file in files:
        data = file.read_bytes()
        header += struct.pack("<32sII", file.name.encode("ascii"), offset, len(data))
        payload += data
        offset += len(data)
    packed = header + payload
    output.write_bytes(struct.pack("<I", len(packed)) + zlib.compress(packed, 9))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pack(args.source, args.output)
