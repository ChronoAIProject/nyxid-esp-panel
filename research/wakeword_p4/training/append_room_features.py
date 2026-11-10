"""Append measured-room training renderings, preserving every existing feature byte."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from stream_data import worker, copy_negative_assets


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--rirs", type=Path, required=True)
    p.add_argument("--rir-audit", type=Path, required=True)
    p.add_argument("--backgrounds", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only-missing", action="store_true",
                   help="Append room variants only for sources that do not already have them")
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    base, root = a.base.resolve(), a.output.resolve()
    audit = json.loads(a.rir_audit.read_text())
    if not audit["all_pinned_hashes_verified"]:
        raise ValueError("Unverified room source")
    rirs = []
    for row in audit["records"]:
        path = (a.rirs/row["path"]).resolve()
        if row["split"] != "training" or not path.is_relative_to(a.rirs.resolve()) or sha(path) != row["sha256"]:
            raise ValueError("Changed or non-training room response")
        rirs.append({**row, "absolute_path": str(path)})
    config = {"rirs": rirs, "snr_db": [-5, 10]}
    index = json.loads((base/"features.json").read_text())
    base_count = len(index)
    offset = sum(r["frames"] for r in index)
    if (base/"features.bin").stat().st_size != offset*80:
        raise ValueError("Base binary size mismatch")
    entries = list(map(json.loads, (base/"dataset/manifest.jsonl").read_text().splitlines()))
    existing = {(r["path"], r["variant"]) for r in index}
    jobs = []
    background_rows = json.loads((a.backgrounds/"manifest.json").read_text())["records"]
    backgrounds = []
    for row in background_rows:
        if row["split"] == "training":
            path = a.backgrounds/row["path"]
            if sha(path) != row["sha256"]:
                raise ValueError("Changed background")
            backgrounds.append(str(path.resolve()))
    for row in entries:
        if row["split"] != "training":
            continue
        for variant in (4, 5):
            if (row["path"], variant) in existing:
                if a.only_missing:
                    continue
                raise ValueError("Requested rendering already exists")
            jobs.append((row, str(base/"dataset"), variant, backgrounds, config))
    root.mkdir(parents=True)
    (root/"dataset").symlink_to(base/"dataset", target_is_directory=True)
    copy_negative_assets(base, root)
    shutil.copyfile(base/"features.bin", root/"features.bin")
    with (root/"features.bin").open("ab") as output, ProcessPoolExecutor(max_workers=a.workers) as pool:
        for number, (record, raw) in enumerate(pool.map(worker, jobs, chunksize=10), 1):
            output.write(raw.astype("<u2").tobytes())
            index.append({**record, "offset": offset, "frames": len(raw)})
            offset += len(raw)
            if number % 1000 == 0:
                print(json.dumps({"rendered": number, "planned": len(jobs)}), flush=True)
    (root/"features.json").write_text(json.dumps(index)+"\n")
    report = {"base_index_sha256": sha(base/"features.json"),
              "base_binary_sha256": sha(base/"features.bin"), "base_sequences": base_count,
              "added_training_sequences": len(jobs), "rir_audit_sha256": sha(a.rir_audit),
              "source_sha256": sha(Path(__file__)), "renderer_sha256": sha(Path(__file__).with_name("stream_data.py")),
              "snr_range_db": config["snr_db"], "variants": [4, 5],
              "policy": "Measured RIR for both variants; colored noise for 4, source-isolated training background for 5. Original evaluation untouched.",
              "limits": "Combined room/noise intervention, not an isolated RIR ablation. Source variants are correlated."}
    (root/"augmentation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
