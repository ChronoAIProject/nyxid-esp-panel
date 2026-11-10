"""Launch Python with pip-installed CUDA 12 runtime paths for TensorFlow.

Usage: python3 gpu_python.py train_stream.py --device cuda ...
Keeps TensorFlow's CUDA 12 paths local to this process tree; PyTorch's CUDA 13
installation and system linker configuration are not modified.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: gpu_python.py SCRIPT [ARGS...]")
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Install tensorflow[and-cuda] first")
    roots = [Path(p) for p in spec.submodule_search_locations]
    libraries, binaries = [], []
    for root in roots:
        for component in sorted(root.iterdir()):
            if component.name == "cu13":
                continue
            if (component / "lib").is_dir():
                libraries.append(str(component / "lib"))
            if (component / "bin").is_dir():
                binaries.append(str(component / "bin"))
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = os.pathsep.join(binaries + [env.get("PATH", "")])
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)


if __name__ == "__main__":
    main()
