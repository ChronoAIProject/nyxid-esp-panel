"""Verify pinned MIT environmental room responses before training augmentation."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi
import numpy as np
import soundfile as sf

REPO = "davidscripka/MIT_environmental_impulse_responses"
REVISION = "b824a1ef2821f112fda0b9cb26e4278c62b425bb"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    records = []
    for entry in info.siblings:
        if not entry.rfilename.startswith("16khz/") or not entry.rfilename.endswith(".wav"):
            continue
        path = a.assets/entry.rfilename
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry.size:
            raise ValueError("RIR size mismatch")
        if entry.lfs:
            if digest != entry.lfs.sha256:
                raise ValueError("Pinned RIR LFS hash mismatch")
        elif hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest() != entry.blob_id:
            raise ValueError("Pinned RIR Git blob hash mismatch")
        pcm, rate = sf.read(path, dtype="float32")
        if rate != 16000 or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all() or not np.any(pcm):
            raise ValueError("Invalid RIR PCM")
        records.append({"path": entry.rfilename, "sha256": digest, "samples": len(pcm),
                        "rate": rate, "peak": float(np.abs(pcm).max()),
                        "energy": float(np.sum(pcm**2)), "split": "training"})
    if not records:
        raise ValueError("No RIRs")
    report = {"repo": REPO, "revision": REVISION, "responses": len(records),
              "all_pinned_hashes_verified": True, "records": records,
              "use": "Training-only real measured room impulse response augmentation; not yet applied to a candidate.",
              "limits": "Response files do not establish independent physical rooms or deployment coverage. No validation/final fixtures were changed."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
