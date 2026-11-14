"""Compare GPU per-row descriptions to original NPZs and the frozen Mac report."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from describe_old_regression_boundaries_mac_v1 import BOUNDARIES, stats


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--isolated-root", type=Path, required=True)
    p.add_argument("--mac-report", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    gpu_path = "research/wakeword_p4/results/4090-old-fixed-grid-regression-boundary-supplement-v1.json"
    gpu_bytes = subprocess.check_output(["git", "-C", str(a.repository_root), "show", "66cb712:" + gpu_path])
    assert sha(gpu_bytes) == "c2d713cc1f5ae987571d714c873097bbd939ff158c28df471c287c0950e0df81"
    gpu = json.loads(gpu_bytes)
    mac_bytes = a.mac_report.read_bytes()
    mac = json.loads(mac_bytes)
    root = a.isolated_root
    metrics = json.loads((root / "research/wakeword_p4/results/4090-old-fixed-grid-regression-metrics-v2.json").read_text())
    manifest_bytes = (root / ".local/wakeword/hey-development-replay-v1.json").read_bytes()
    assert sha(manifest_bytes) == mac["synthetic_manifest_sha256"]
    records = json.loads(manifest_bytes)["records"]
    key = lambda row: (row["model"], row["kind"], row["threshold"])
    gpu_points = {key(row): row for row in gpu["points"]}
    mac_points = {key(row): row for row in mac["points"]}
    metric_points = {key(row): row for row in metrics["points"]}
    assert len(gpu_points) == len(mac_points) == len(metric_points) == 1242
    assert gpu_points.keys() == mac_points.keys() == metric_points.keys()
    rows_checked = boundaries_checked = 0
    for k, point in gpu_points.items():
        original = metric_points[k]
        m = mac_points[k]
        b = (root / original["events_path"]).read_bytes()
        assert sha(b) == original["events_sha256"] == point["events_sha256"] == m["original_events_sha256"]
        with np.load(root / original["events_path"], allow_pickle=False) as saved:
            offsets, steps = saved["synthetic_offsets"], saved["synthetic_steps"]
            expected_rows = []
            for i, row in enumerate(records):
                if row["target_id"] != m["target"]:
                    continue
                times = [min((3 * int(step) + 2) * 160, row["samples"])
                         for step in steps[offsets[i]:offsets[i + 1]]]
                boundaries = {}
                for boundary in BOUNDARIES:
                    later = [time for time in times if time >= row[boundary]]
                    boundaries[boundary] = dict(
                        first_event_signed_ms=(times[0] - row[boundary]) / 16 if times else None,
                        first_at_or_after_ms=(later[0] - row[boundary]) / 16 if later else None)
                    boundaries_checked += 1
                expected_rows.append(dict(index=i, source_id=row["source_id"], event_count=len(times), boundaries=boundaries))
            assert expected_rows == point["positive_rows"], k
            assert len(expected_rows) == 36
            rows_checked += len(expected_rows)
        for boundary in BOUNDARIES:
            values = [row["boundaries"][boundary]["first_at_or_after_ms"] for row in point["positive_rows"]]
            assert stats([v for v in values if v is not None]) == m["boundaries"][boundary]["first_at_or_after_boundary_signed_ms"]
    ranges = []
    for model, kind in sorted({(r["model"], r["kind"]) for r in mac["points"]}):
        selected = [r for r in mac["points"] if (r["model"], r["kind"]) == (model, kind)]
        assert len(selected) == 138
        ranges.append(dict(model=model, kind=kind, points=138, counts={name: dict(
            minimum=min(r["counts"][name] for r in selected), maximum=max(r["counts"][name] for r in selected))
            for name in selected[0]["counts"]}))
    report = dict(schema="mac-old-regression-boundary-ledger-cross-audit-v1",
                  script_sha256=sha(Path(__file__).read_bytes()), mac_report_sha256=sha(mac_bytes),
                  gpu_boundary_report_sha256=sha(gpu_bytes), points=1242,
                  original_npz_positive_rows_checked=rows_checked, per_row_boundaries_checked=boundaries_checked,
                  all_gpu_positive_row_descriptions_exact=True, all_mac_aggregate_complete_boundary_statistics_exact=True,
                  all_original_prefix_metrics_preserved=mac["original_prefix_first_hit_statistics_exact"],
                  descriptive_count_ranges=ranges, selection_authorized=False,
                  limits="Original consumed old four models plus v30 only. Both per-row first-event/complete-boundary GPU descriptions and Mac complete-boundary aggregates agree with original authenticated NPZs. Mac first-after-source and GPU first-event intentionally have different definitions. No inference, new events, threshold selection, current bounded-family evaluation, fresh development/acceptance or promotion.")
    with a.report.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "descriptive_count_ranges"}))
    for row in ranges:
        print(json.dumps({k: row[k] for k in ["model", "kind"]} | {"any_event_clips": row["counts"]["any_event_clips"]}))


if __name__ == "__main__":
    main()
