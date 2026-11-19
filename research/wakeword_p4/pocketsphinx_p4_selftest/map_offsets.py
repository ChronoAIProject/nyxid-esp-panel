"""Locate unique complete acoustic files in the exact raw FatFs image being flashed."""
import hashlib
import json
import sys
from pathlib import Path


def generate(image_path, work, header):
    image = image_path.read_bytes()
    expected = json.loads((work / "inputs.json").read_bytes())["files"]
    rows = []
    for name in ("am/mdef", "am/sendump"):
        data = (work / "files" / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected[name]:
            raise ValueError(f"input hash mismatch for {name}")
        offset = image.find(data)
        if not data or offset < 0 or image.find(data, offset + 1) >= 0:
            raise ValueError(f"expected one contiguous copy of {name}")
        rows.append(f'    {{"/vol/{name}", {offset}, {len(data)}}},\n')
    content = "".join(rows)
    if not header.exists() or header.read_text() != content:
        header.write_text(content)


if __name__ == "__main__":
    generate(*map(Path, sys.argv[1:]))
