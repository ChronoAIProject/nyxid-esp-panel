"""Independent JSON-only audit of original official event/boundary descriptions."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

GPU_COMMIT = "8a4eea3920aa8fd2db7696e5549c3eb4906fa861"
FILES = {
    "results/4090-original-official-development-calibration-v1.json": "04d7eeb55cac54fc3324c4c9ce1e73e4efb8d44d8852ae90fb0380483fcdafe6",
    "results/4090-hey-official-development-execution-v1.json": "889e6333cc86aaf6e559c71cd423701ab458ab0b69841c008b4f1a168a6d9060",
    "results/4090-official-boundary-event-descriptive-v1.json": "b7fb53600ffdba5bb643d1e599cbf11589a7e2872f3e4d8fdfb6496816a5ce5e",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    raw = {name: subprocess.check_output(["git", "show", GPU_COMMIT + ":research/wakeword_p4/" + name], cwd=a.repository_root) for name in FILES}
    assert all(hashlib.sha256(data).hexdigest() == FILES[name] for name, data in raw.items())
    calibration, execution, descriptive = [json.loads(raw[name]) for name in FILES]
    assert calibration["execution_sha256"] == FILES["results/4090-hey-official-development-execution-v1.json"]
    assert not any(calibration[k] for k in ["threshold_selected", "acceptance_authorized", "candidate_promoted"])
    assert not descriptive["selection_authorized"]
    assert all(h in descriptive["files"].values() for h in [FILES[k] for k in list(FILES)[:2]])
    rows = [r for r in execution["records"] if r["phase"] == "synthetic"]
    background = [r for r in execution["records"] if r["phase"] != "synthetic"]
    assert len(rows) == 360 and len(background) == 12
    source = {(r["source_id"], r["variant"]): r for r in rows}
    assert len(source) == 360
    seconds = sum(r["samples"] for r in background) / 16000
    points, audited_cases, positive_events = [], 0, 0
    for model, curve in calibration["curves"].items():
        assert len(curve) == 13
        target = {"willow": "hey_willow", "hiesp": "hi_esp"}[model]
        for point in curve:
            original = next(p for p in descriptive["points"] if p["model"] == model and p["request"] == point["request"])
            assert original["actual_float32_bits"] == point["actual_float32_bits"]
            assert len(point["synthetic"]["records"]) == 360
            assert [(r["source_id"], r["variant"]) for r in point["synthetic"]["records"]] == [(r["source_id"], r["variant"]) for r in rows]
            counts = Counter(no_event=0, pre_source_only=0, after_source_before_prefix_only=0,
                             complete_phrase=0, after_source_any=0, pre_prefix_and_complete=0)
            descriptions = {(r["source_id"], r["variant"]): r for r in original["cases"]}
            assert len(descriptions) == 36
            for record in point["synthetic"]["records"]:
                row = source[record["source_id"], record["variant"]]
                assert row["target_id"] == record["target_id"]
                if row["target_id"] != target:
                    continue
                described = descriptions[record["source_id"], record["variant"]]
                times = []
                for event in record["events"]:
                    sample = event["sample_end"]
                    assert type(sample) is int and sample == event["call"] * 512 and event["result"] > 0
                    assert 0 < sample <= math.ceil(row["samples"] / 512) * 512
                    for boundary in ["generated_prefix_end_sample", "full_rir_prefix_end_sample",
                                     "generated_container_end_sample", "full_rir_container_end_sample"]:
                        assert event[boundary + "_signed_ms"] == (sample - row[boundary]) / 16
                    times.append(sample)
                assert times == sorted(times)
                before = any(row["lead_samples"] <= t < row["generated_prefix_end_sample"] for t in times)
                complete = any(t >= max(row["lead_samples"], row["generated_prefix_end_sample"]) for t in times)
                after_source = any(t >= row["lead_samples"] for t in times)
                category = "complete_phrase" if complete else "after_source_before_prefix_only" if after_source else "pre_source_only" if times else "no_event"
                counts[category] += 1
                counts["after_source_any"] += after_source
                counts["pre_prefix_and_complete"] += before and complete
                assert described["category"] == category and described["events"] == record["events"]
                assert described["carrier"] == record["carrier"]
                assert all(described[k] == row[k] for k in ["lead_samples", "generated_prefix_end_sample", "full_rir_prefix_end_sample"])
                positive_events += len(times)
                audited_cases += 1
            assert dict(counts) == original["counts"]
            assert counts["complete_phrase"] == point["synthetic"]["complete_phrase_activated"]
            assert counts["after_source_any"] / 36 == point["synthetic"]["any_after_source_recall"]
            assert math.isclose(point["background_hours"] * 3600, seconds, abs_tol=1e-7)
            points.append(dict(model=model, request=point["request"], actual_float32_bits=point["actual_float32_bits"],
                               original_positive_clips=36, counts=dict(counts),
                               original_background_events=point["background_events"],
                               original_background_upper95_per_hour=point["background_upper95"],
                               original_eligible_diagnostic=point["eligible_diagnostic"]))
    assert len(points) == 26 and audited_cases == 936
    result = dict(schema="mac-official-boundary-events-independent-audit-v1", gpu_commit=GPU_COMMIT,
                  source_sha256=FILES, points=points, positive_case_evaluations=audited_cases,
                  original_positive_events_checked=positive_events, background_samples=sum(r["samples"] for r in background),
                  original_event_sample_clocks_and_signed_boundaries_verified=True, all_passed=True,
                  model_invocations=0, audio_decodes=0, thresholds_selected=False, acceptance_authorized=False,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limits="JSON-only independent checks of saved event descriptions, categories and source contracts at all original 26 points. Original calibration/raw inference not reproduced; raw per-call UART streams not independently re-audited here. Background counts/confidence bounds copied from original report, not independently recomputed. Generated boundaries are synthesis allocations, not measured phonetic ends. Earlier events are observed triggers, not proof of complete-word recognition; no counterfactual refractory claim, rescoring, threshold selection or recognition qualification.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(points=26, positive_cases=audited_cases, positive_events=positive_events,
                          all_passed=True, model_invocations=0)))


if __name__ == "__main__":
    main()
