"""Describe persistent errors across every frozen point of audited old data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ranges(points):
    if not points:
        return None
    return {key: [min(p["positive"]["all"][key] for p in points), max(p["positive"]["all"][key] for p in points)]
            for key in ["any_after_source", "complete", "before_only", "missing"]} | {
        "common_false_clips": [min(p["negative"]["common"]["activated_clips"] for p in points),
                               max(p["negative"]["common"]["activated_clips"] for p in points)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--intake-report", type=Path, required=True)
    p.add_argument("--array-audit", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    intake_bytes, audit_bytes = a.intake_report.read_bytes(), a.array_audit.read_bytes()
    intake, audit = json.loads(intake_bytes), json.loads(audit_bytes)
    assert intake["all_passed"] and audit["all_passed"] and audit["points"] == 552
    assert intake["gpu_commit"] == audit["gpu_commit"]
    mb = (a.packet_root / "manifest.json").read_bytes()
    assert sha(mb) == intake["manifest_sha256"]
    manifest = json.loads(mb)

    def read(name):
        data = (a.packet_root / name).read_bytes()
        assert sha(data) == manifest["files"][name]["sha256"]
        return data

    rows = json.loads(read("materials/synthetic.json"))["records"]
    source = {(r["source_id"], r["variant"]): r for r in rows}
    assert len(source) == len(rows) == 360
    model_reports = []
    for model in audit["models"]:
        name = model["model"]
        data = read("diagnostic/" + name + ".jsonl")
        assert sha(data) == model["original_jsonl_sha256"]
        points = [json.loads(line) for line in data.splitlines()]
        target = name.split("-", 1)[0]
        own = {key for key, row in source.items() if row["target_id"] == target}
        common = {key for key, row in source.items() if row["target_id"] not in ["hey_willow", "hey_nyx"]}
        assert len(own) == 36 and len(common) == 288
        false_sets, miss_sets = [], []
        for point in points:
            false, miss = set(), set()
            for row in point["records"]:
                if row["phase"] != "synthetic":
                    continue
                key = (row["source_id"], row["variant"])
                if key in common and row["events"]:
                    false.add(key)
                if key in own and not row["positive"]["any_after_source"]:
                    miss.add(key)
            false_sets.append(false)
            miss_sets.append(miss)
        persistent = set.intersection(*false_sets)
        persistent_miss = set.intersection(*miss_sets)
        far_region = [p["summary"] for p in points if p["summary"]["background"]["far_upper95"] <= .5]
        persistent_rows = [{k: source[key][k] for k in ["source_id", "variant", "literal", "text", "carrier", "length_scale", "voice"]}
                           for key in sorted(persistent)]
        model_reports.append(dict(model=name, registered_points=len(points), all_point_ranges=ranges([p["summary"] for p in points]),
                                  original_far_upper95_le_half_region_points=len(far_region), far_region_ranges=ranges(far_region),
                                  false_clips_at_every_registered_point=len(persistent),
                                  persistent_common_false_by_literal=dict(sorted(Counter(source[k]["literal"] for k in persistent).items())),
                                  persistent_common_false_records=persistent_rows,
                                  own_clips_missing_at_every_registered_point=len(persistent_miss),
                                  persistent_miss_by_carrier=dict(sorted(Counter(source[k]["carrier"] for k in persistent_miss).items())),
                                  points_with_zero_common_false=sum(not x for x in false_sets)))
    result = dict(schema="mac-fp32-reused-all-point-persistent-errors-v1", gpu_commit=audit["gpu_commit"],
                  intake_report_sha256=sha(intake_bytes), array_audit_sha256=sha(audit_bytes),
                  synthetic_manifest_sha256=sha(read("materials/synthetic.json")),
                  observed_voices=sorted({r["voice"] for r in rows}),
                  observed_source_speed_values=sorted({r["length_scale"] for r in rows}),
                  background_samples=audit["background_samples"], background_hours=audit["background_hours"],
                  models=model_reports, model_invocations=0, media_reads=0,
                  thresholds_selected=False, candidate_promoted=False,
                  source_sha256=sha(Path(__file__).read_bytes()),
                  limits="Intersections/ranges over every original fixed-grid point on already-used development. No selected threshold, new inference, audio pronunciation verification, independent acceptance or causal proof. Variants/carriers/speed versions from one TTS voice are not independent speakers. Persistent confusions establish an observed discrimination problem, not a unique training/data root cause.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"observed_voices": result["observed_voices"], "models": [{k: m[k] for k in ["model", "persistent_common_false_by_literal", "own_clips_missing_at_every_registered_point"]} for m in model_reports]}))


if __name__ == "__main__":
    main()
