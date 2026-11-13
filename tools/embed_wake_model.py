#!/usr/bin/env python3
"""Pack a pinned ESP-SR WakeNet9 model into the application OTA image."""
import argparse
from pathlib import Path
import struct
import zlib


MODELS = ("wn9_hiesp", "wn9_heywillow_tts")


def pack(source: Path, output: Path, model_name="wn9_hiesp"):
    if model_name not in MODELS or source.name != model_name:
        raise ValueError("Source directory must match the explicitly selected supported model")
    files = sorted(p for p in source.iterdir() if p.is_file())
    if {p.name for p in files} != {"_MODEL_INFO_", "wn9_index", "wn9_data"}:
        raise ValueError("Unexpected WakeNet9 model files")
    offset = 4 + 36 + 40 * len(files)
    header = struct.pack("<I32sI", 1, model_name.encode("ascii"), len(files))
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
    parser.add_argument("--model-name", choices=MODELS, default="wn9_hiesp")
    args = parser.parse_args()
    pack(args.source, args.output, args.model_name)
