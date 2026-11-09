"""Launch training from a private immutable source copy in a fresh run directory.

Pass the normal train_stream.py options. This prevents edits to the research
checkout from changing TensorFlow's inspected function source during a run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    snapshot = root/"source"
    snapshot.mkdir()
    source = Path(__file__).resolve().parent
    hashes = {}
    for path in sorted(source.glob("*.py")):
        data = path.read_bytes()
        (snapshot/path.name).write_bytes(data)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    for path in source.glob("requirements-*.lock"):
        shutil.copyfile(path, snapshot/path.name)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source.parents[2],
                              text=True, capture_output=True)
    command = [sys.executable, str(snapshot/"train_stream.py"), "--output", str(root), *remaining]
    (root/"launch.json").write_text(json.dumps({"source_files_sha256": hashes,
        "checkout_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "command": command, "scope": "Private research source snapshot; no environment/secrets copied"}, indent=2)+"\n")
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
