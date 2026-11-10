"""Training-only consistency subset; preserve all parent WAVs and original labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--asr-details", type=Path, required=True)
    p.add_argument("--asr-report", type=Path, required=True)
    p.add_argument("--exclude-manifest", type=Path, help="Emit only new source rows not already retained")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a fresh output directory")
    if not (a.dataset/"generation.json").is_file():
        raise ValueError("Incomplete synthesis")
    rows = list(map(json.loads, (a.dataset/"manifest.jsonl").read_text().splitlines()))
    details = list(map(json.loads, a.asr_details.read_text().splitlines()))
    audit = json.loads(a.asr_report.read_text())
    if not audit["training_source"] or audit["manifest_sha256"] != sha(a.dataset/"manifest.jsonl"):
        raise ValueError("ASR audit does not match training manifest")
    indexed = {r["path"]: r for r in details}
    if len(indexed) != len(details) or set(indexed) != {r["path"] for r in rows}:
        raise ValueError("ASR details do not cover each source exactly once")
    previous_rows = list(map(json.loads, a.exclude_manifest.read_text().splitlines())) if a.exclude_manifest else []
    previous = {r["path"]: r["sha256"] for r in previous_rows}
    previous_hashes = set(previous.values())
    selected, excluded = [], []
    for row in rows:
        detail = indexed[row["path"]]
        path = (a.dataset/row["path"]).resolve()
        if row["split"] != "training" or not path.is_relative_to(a.dataset.resolve()):
            raise ValueError("Only training-only source WAVs may be selected")
        if sha(path) != row["sha256"] or detail["sha256"] != row["sha256"] or detail["expected"] != row["text"]:
            raise ValueError("Changed audio or expected text")
        if row["path"] in previous:
            if previous[row["path"]] != row["sha256"]:
                raise ValueError("Existing source path changed")
            excluded.append({"path": row["path"], "label": row["label"], "sha256": row["sha256"], "reason": "Already retained; preserve existing corpus"})
        elif detail["text_agreement"] is True:
            if row["sha256"] in previous_hashes:
                raise ValueError("New path duplicates previously retained WAV")
            selected.append(row)
        else:
            excluded.append({"path": row["path"], "label": row["label"],
                             "sha256": row["sha256"], "reason": "ASR text disagreement"})
    if {r["label"] for r in selected} != {0, 1}:
        raise ValueError("Selected source needs both classes")
    a.output.mkdir(parents=True)
    for row in selected:
        destination = a.output/row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.dataset/row["path"], destination)
    (a.output/"manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in selected))
    report = {"exclude_manifest_sha256": sha(a.exclude_manifest) if a.exclude_manifest else None,
              "previous_rows": len(previous),
              "source_sha256": sha(Path(__file__)),
              "parent_manifest_sha256": sha(a.dataset/"manifest.jsonl"),
              "parent_generation_sha256": sha(a.dataset/"generation.json"),
              "asr_report_sha256": sha(a.asr_report), "asr_details_sha256": sha(a.asr_details),
              "selected": len(selected), "selected_positives": sum(r["label"] for r in selected),
              "selected_negatives": sum(not r["label"] for r in selected), "excluded": excluded,
              "manifest_sha256": sha(a.output/"manifest.jsonl"),
              "labels_changed": 0, "wake_scores_used": False,
              "policy": "Keep exact independently audited ASR text agreement (including declared positive aliases); training only.",
              "limits": "ASR consistency is not phonetic ground truth. Selection may bias accents; all original audio remains intact. No evaluation data selected or filtered."}
    (a.output/"generation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
