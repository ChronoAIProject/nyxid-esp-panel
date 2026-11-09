"""Calibrate quantized V2 on validation, then report regression measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2
import soundfile as sf

from evaluate import Detector, THRESHOLDS, events, smooth
from features import features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--mac-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true",
                        help="Calibrate on validation without reading test or Mac regression audio")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    assert not (root/"report.json").exists(), "Use a fresh evaluation directory"
    assert not (root/"threshold.json").exists(), "Do not overwrite an existing calibration"
    detector = Detector(args.model)
    raw = np.memmap(args.features/"features.bin", mode="r", dtype="<u2").reshape(-1, 40)
    records = json.loads((args.features/"features.json").read_text())
    ambient_rows = json.loads((args.features/"human.json").read_text())
    ambient_rows += json.loads((args.features/"ambient.json").read_text())

    def measure(split):
        clips, ambient = [], []
        for row in records:
            if row["split"] != split:
                continue
            detector.reset()
            values = smooth(detector.predict(raw[row["offset"]:row["offset"]+row["frames"]]))
            times = (3*(np.arange(len(values))+4)+5)*.01
            end = (row["keyword_frame"]+2)*.01 if row["label"] else None
            valid = (times >= end-.10) & (times <= end+.80) if row["label"] else np.ones(len(values), dtype=bool)
            clips.append({"source": row["path"], "engine": row["engine"], "variant": row["variant"],
                          "label": row["label"], "text": row["text"], "max_score": float(values[valid].max()) if valid.any() else 0,
                          "scores": values.tolist(), "times": times.tolist(), "keyword_end": end})
        for number, row in enumerate(ambient_rows):
            if row["split"] != split:
                continue
            data = np.load(args.features/row["path"], mmap_mode="r")
            detector.reset()
            values = smooth(detector.predict(data))
            np.save(root/f"ambient-{split}-{number:03d}.npy", values)
            ambient.append({"hours": len(data)*.01/3600, "events": {str(t): events(values, t) for t in THRESHOLDS},
                            "source_type": "speech" if "speaker" in row else "environment"})
        return clips, ambient

    def summarize(clips, ambient, threshold):
        pos = [r for r in clips if r["label"]]
        neg = [r for r in clips if not r["label"]]
        hours = sum(r["hours"] for r in ambient)
        count = sum(r["events"][str(threshold)] for r in ambient)
        recalls = {e: float(np.mean([r["max_score"] >= threshold for r in pos if r["engine"] == e]))
                   for e in {r["engine"] for r in pos}}
        false_rates = {e: float(np.mean([r["max_score"] >= threshold for r in neg if r["engine"] == e]))
                       for e in {r["engine"] for r in neg}}
        return {"threshold": threshold, "positive_variants": len(pos),
                "recall": float(np.mean([r["max_score"] >= threshold for r in pos])),
                "macro_recall": float(np.mean(list(recalls.values()))), "recall_by_engine": recalls,
                "hard_negative_variants": len(neg),
                "hard_negative_false_rate": float(np.mean([r["max_score"] >= threshold for r in neg])),
                "max_source_false_rate": max(false_rates.values()),
                "ambient_hours": hours, "ambient_false_events": count, "ambient_false_per_hour": count/hours,
                "ambient_false_per_hour_upper_95": float(.5*chi2.ppf(.95, 2*(count+1))/hours)}

    val, va = measure("validation")
    grid = [summarize(val, va, t) for t in THRESHOLDS]
    eligible = [r for r in grid if r["ambient_false_per_hour"] <= .5 and r["max_source_false_rate"] <= .01]
    selected = max(eligible, key=lambda r: (r["macro_recall"], -r["ambient_false_per_hour"])) if eligible else min(
        grid, key=lambda r: (r["ambient_false_per_hour"], r["max_source_false_rate"], -r["macro_recall"]))
    (root/"threshold.json").write_text(json.dumps({"selected": selected, "grid": grid,
                                                  "viable_operating_point": bool(eligible)}, indent=2)+"\n")
    print(json.dumps({"validation": selected, "viable_operating_point": bool(eligible)}), flush=True)
    if args.validation_only:
        (root/"validation-details.json").write_text(json.dumps({"validation": val})+"\n")
        return
    # Quantized threshold is frozen before regression/test data is evaluated.
    test, ta = measure("testing")
    threshold = selected["threshold"]
    mac = []
    for item in map(json.loads, (args.mac_eval/"manifest.jsonl").read_text().splitlines()):
        pcm, rate = sf.read(args.mac_eval/item["path"], dtype="int16")
        assert rate == 16000
        for variant in (0, 1):
            wave = np.concatenate((np.zeros(28800), pcm, np.zeros(16000))).astype(np.float32)
            if variant:
                rng = np.random.default_rng(int(hashlib.sha256(item['path'].encode()).hexdigest()[:8], 16))
                wave = wave*.3+rng.normal(0, 300, len(wave))
            detector.reset()
            values = smooth(detector.predict(features(wave)))
            mac.append({**item, "variant": variant, "max_score": float(values.max())})
    mp = [r for r in mac if r["label"]]
    mn = [r for r in mac if not r["label"]]
    report = {"model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "sliding_window": 5, "threshold": threshold, "validation": selected,
              "testing_regression": summarize(test, ta, threshold),
              "mac_regression": {"positive_variants": len(mp), "negative_variants": len(mn),
                                 "recall": float(np.mean([r["max_score"] >= threshold for r in mp])),
                                 "false_rate": float(np.mean([r["max_score"] >= threshold for r in mn]))},
              "by_engine": {e: summarize([r for r in test if r["engine"] == e], ta, threshold)
                            for e in sorted({r["engine"] for r in test})},
              "limits": ["v1 test/Mac sets informed development; these are regression results, not untouched final holdouts.",
                         "Four render variants share source utterances and are not independent trial counts.",
                         "MS-SNSD files are blob-disjoint; original room/recording session identities are unavailable.",
                         "No real-user positive wake-word recordings or Chinese/TV/music negative evaluation.",
                         "Not production acceptance; confidence bounds apply only to the measured audio distribution."]}
    (root/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    (root/"clip-details.json").write_text(json.dumps({"validation": val, "testing": test, "mac": mac})+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
