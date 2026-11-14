"""Package only the candidate-free probe's original artifacts and source."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    run = a.run_root
    files = {"original/" + name: run / name for name in [
        "model.tflite", "synthetic-arrays.npz", "export-result.json", "elf-section-sizes.txt",
        "project/CMakeLists.txt", "project/main/CMakeLists.txt", "project/main/expected.h",
        "project/sdkconfig.defaults", "project/sdkconfig", "build/compile_commands.json",
        "build/nyx_fp32_feasibility.elf", "build/nyx_fp32_feasibility.bin", "build/nyx_fp32_feasibility.map"]}
    for name in ["build.log", "configure-v2.log", "build-v2.log"]:
        files["original/" + name] = run.with_name(run.name + "-" + name)
    files["original/export.log"] = run.with_name(run.name + ".log")
    prefix = "research/wakeword_p4/"
    for name in ["experiments/probe_fp32_runtime_feasibility_mac_v1.py",
                 "experiments/fp32_runtime_feasibility_p4_v1.cpp",
                 "experiments/collect_fp32_feasibility_build_mac_v1.py",
                 "experiments/package_fp32_feasibility_artifact_mac_v1.py",
                 "experiments/fp32_runtime_feasibility_mac_v1.md",
                 "results/mac-fp32-runtime-feasibility-plan-v1.json",
                 "results/mac-fp32-runtime-feasibility-export-v1.json",
                 "results/mac-fp32-runtime-feasibility-p4-link-v1.json"]:
        files["source/" + prefix + name] = a.repository_root / (prefix + name)
    data = {name: path.read_bytes() for name, path in files.items()}
    digest = lambda b: hashlib.sha256(b).hexdigest()
    manifest = dict(schema="mac-fp32-runtime-feasibility-artifact-v1",
                    scope="Seeded random weights/synthetic features and original full P4 link only; no board, trained model, audio or recognition evidence",
                    files={name: dict(bytes=len(b), sha256=digest(b)) for name, b in sorted(data.items())})
    data["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with a.archive.open("xb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as z, tarfile.open(fileobj=z, mode="w") as t:
        for name, b in sorted(data.items()):
            info = tarfile.TarInfo(name); info.size = len(b); info.mode = 0o644
            t.addfile(info, io.BytesIO(b))
    packed = a.archive.read_bytes()
    receipt = dict(schema="mac-fp32-runtime-feasibility-artifact-v1", archive=a.archive.name,
                   bytes=len(packed), sha256=digest(packed), entries=len(data), payloads=len(files),
                   manifest_sha256=digest(data["manifest.json"]), candidate_qualified=False,
                   public_url="https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/" + a.archive.name,
                   anonymous_redownload_verified=False)
    with a.receipt.open("x") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
