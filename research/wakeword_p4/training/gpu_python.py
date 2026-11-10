"""Launch Python with process-local pip NVIDIA runtime paths.

Usage: python3 gpu_python.py train_stream.py --device cuda ...
Use --cuda-major 13 before SCRIPT for CUDA 13 ONNX Runtime. The default
remains CUDA 12 for TensorFlow; system linker configuration is not modified.
"""
import importlib.util
import os
from pathlib import Path
import sys


def main():
    arguments = sys.argv[1:]
    major = "12"
    if arguments[:1] == ["--cuda-major"]:
        if len(arguments) < 3 or arguments[1] not in ("12", "13"):
            raise SystemExit("--cuda-major requires 12 or 13 followed by SCRIPT")
        major, arguments = arguments[1], arguments[2:]
    if not arguments:
        raise SystemExit("Usage: gpu_python.py SCRIPT [ARGS...]")
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("Install tensorflow[and-cuda] first")
    roots = [Path(p) for p in spec.submodule_search_locations]
    libraries, binaries = [], []
    for root in roots:
        for component in sorted(root.iterdir()):
            if component.name == "cu13" and major == "12":
                continue
            if (component / "lib").is_dir():
                libraries.append(str(component / "lib"))
            if (component / "bin").is_dir():
                binaries.append(str(component / "bin"))
    if major == "13":
        libraries.sort(key=lambda path: 0 if "/cu13/" in path else 1)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    env["PATH"] = os.pathsep.join(binaries + [env.get("PATH", "")])
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.execve(sys.executable, [sys.executable, *arguments], env)


if __name__ == "__main__":
    main()
