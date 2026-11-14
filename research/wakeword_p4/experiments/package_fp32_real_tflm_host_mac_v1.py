"""Package the original candidate-free real-TFLM host run without rerunning it."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["repository-root", "run-root", "prior-root", "archive", "receipt"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    files = {"original-host/" + name: a.run_root / name for name in [
        "report.json", "compile.json", "execution.json", "inputs.bin", "outputs.bin", "saved-arrays.npz", "fp32-real-tflm", "esp_timer.h"]}
    files.update({"original-export/" + name: a.prior_root / name for name in ["model.tflite", "synthetic-arrays.npz", "export-result.json"]})
    files["original-host/log.txt"] = a.run_root.with_name(a.run_root.name + ".log")
    prefix = "research/wakeword_p4/"
    for path in ["experiments/fp32_real_tflm_host_v1.cpp", "experiments/run_fp32_real_tflm_host_mac_v1.py",
                 "experiments/audit_fp32_real_tflm_host_mac_v1.py", "experiments/package_fp32_real_tflm_host_mac_v1.py",
                 "experiments/fp32_real_tflm_host_mac_v1.md", "results/mac-fp32-real-tflm-host-plan-v1.json",
                 "results/mac-fp32-real-tflm-host-result-v1.json", "results/mac-fp32-real-tflm-host-saved-byte-audit-v1.json"]:
        files["source/" + prefix + path] = a.repository_root / (prefix + path)
    data = {name: path.read_bytes() for name, path in files.items()}
    digest = lambda b: hashlib.sha256(b).hexdigest()
    manifest = dict(schema="mac-fp32-real-tflm-host-artifact-v1", scope="Original random model/synthetic inputs and actual host TFLM outputs; no board or trained-model/recognition evidence",
                    files={name: dict(bytes=len(b), sha256=digest(b)) for name, b in sorted(data.items())})
    data["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with a.archive.open("xb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0) as z, tarfile.open(fileobj=z, mode="w") as t:
        for name, b in sorted(data.items()):
            info = tarfile.TarInfo(name); info.size = len(b); info.mode = 0o644
            t.addfile(info, io.BytesIO(b))
    packed = a.archive.read_bytes()
    receipt = dict(schema=manifest["schema"], archive=a.archive.name, bytes=len(packed), sha256=digest(packed),
                   entries=len(data), payloads=len(files), manifest_sha256=digest(data["manifest.json"]),
                   public_url="https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/research-hey-nyx-20261110/" + a.archive.name,
                   anonymous_redownload_verified=False, p4_execution=False, candidate_qualified=False)
    with a.receipt.open("x") as f: f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
