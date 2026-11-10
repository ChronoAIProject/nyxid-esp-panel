"""Fetch a pinned, hash-checked CC0 subset of MS-SNSD environmental noises."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import soundfile as sf

REVISION = "fe61c4ba0d9ac8dd7e23d719cc79f8947e1dc742"
# Exclude the DEMAND location recordings (Office/Park/Restaurant/etc.). The
# upstream README identifies its other Freesound environmental noises as CC0.
CATEGORIES = ("AirConditioner_", "ShuttingDoor_", "Typing_",
              "VacuumCleaner_", "WasherDryer_", "SqueakyChair_")


def fetch(url):
    with urllib.request.urlopen(url, timeout=45) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tree = json.loads(fetch(f"https://api.github.com/repos/microsoft/MS-SNSD/git/trees/{REVISION}?recursive=1"))
    unique = set()
    rows = []
    # Prioritize original training files when identical blobs occur in both.
    for item in sorted(tree["tree"], key=lambda r: (not r["path"].startswith("noise_train/"), r["path"])):
        path = Path(item["path"])
        if (path.parent.name not in ("noise_train", "noise_test")
                or path.suffix != ".wav" or not path.name.startswith(CATEGORIES)
                or item["sha"] in unique):
            continue
        unique.add(item["sha"])
        split = "testing" if path.parent.name == "noise_test" else "training"
        rows.append({"path": str(path), "blob_sha1": item["sha"], "split": split})
    train = [r for r in rows if r["split"] == "training"]
    ids = np.random.default_rng(726).permutation(len(train))
    for index in ids[:max(1, len(train)//5)]:
        train[index]["split"] = "validation"

    def download(row):
        path = args.output / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/microsoft/MS-SNSD/{REVISION}/{row['path']}"
        data = path.read_bytes() if path.exists() else fetch(url)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != row["blob_sha1"]:
            raise ValueError(f"Git blob mismatch: {row['path']}")
        path.write_bytes(data)
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"Expected 16 kHz mono: {row['path']}")
        return {**row, "sha256": hashlib.sha256(data).hexdigest(),
                "seconds": info.duration, "source": url, "license": "CC0-1.0"}

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(download, rows))
    (args.output / "manifest.json").write_text(json.dumps({
        "revision": REVISION, "records": rows,
        "limits": "File/blob disjoint; original recording-session identities are unavailable. No claim of room/session-disjoint evaluation."
    }, indent=2) + "\n")
    print(json.dumps({s: {"files": sum(r["split"] == s for r in rows),
                              "hours": sum(r["seconds"] for r in rows if r["split"] == s)/3600}
                      for s in ("training", "validation", "testing")}), flush=True)


if __name__ == "__main__":
    main()
