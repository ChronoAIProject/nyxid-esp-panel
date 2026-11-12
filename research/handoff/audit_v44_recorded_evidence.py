"""Check pinned v44 report bytes, recorded choices and gate arithmetic only.

Run from the repository after fetching research-hye-nyx-voice-gpu. This does
not load models, score arrays, waveforms, unused development or final data.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


REF = "75cdd0d25a6fce96acab50e40dc97278d138e548"
PREFIX = "research/wakeword_p4/"
MODEL = "88de16e71b5f49b5e85d226b74a07aa036324dcaab48e6f58c9c8a9343608652"
THRESHOLDS = [.50, .70, .80, .90, .95, .97, .98, .985, .99, .992, .994]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(relative):
    return subprocess.check_output(["git", "show", REF + ":" + PREFIX + relative])


def upper95(events, hours):
    # Invert the integer Poisson CDF independently of the GPU's scipy chi2.
    def cdf(mean):
        term = total = math.exp(-mean)
        for count in range(1, events + 1):
            term *= mean / count
            total += term
        return total
    low, high = 0., max(1., events + 1.)
    while cdf(high) > .05:
        high *= 2
    for _ in range(100):
        mid = (low + high) / 2
        if cdf(mid) > .05:
            low = mid
        else:
            high = mid
    return (low + high) / 2 / hours


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required")
    names = {
        "training": "4090-v44-training-v1.json",
        "prior": "4090-v43-training.json",
        "export": "4090-v44-export-audit-v1.json",
        "validation": "4090-v44-validation-v1.json",
        "phone": "4090-v44-phone-development-v1.json",
        "vctk": "4090-v44-vctk-development-v1.json",
        "background": "4090-v44-voxconverse-development-v1.json",
        "calibration": "4090-v44-reused-development-exploratory-calibration-v1.json",
        "receipt": "4090-v44-score-receipt-v1.json",
        "rejection": "4090-v44-rejection-v1.json",
    }
    raw = {key: blob("results/" + name) for key, name in names.items()}
    reports = {key: json.loads(value) for key, value in raw.items()}
    training, prior, export = (reports[k] for k in ("training", "prior", "export"))
    require(sha(raw["training"]) == export["training_sha256"], "Training hash differs")
    require(training["model_sha256"] == export["model_sha256"] == MODEL, "Export model differs")
    require(export["all_saved_weights_finite"] and export["frozen_bn_unchanged"], "Export audit failed")
    require(training["batchnorm_initial"] == training["batchnorm_selected"], "BN metadata changed")
    require(export["steps"] == training["steps"] == 120000, "Incomplete training report")
    for name, digest in training["source_files_sha256"].items():
        require(Path(name).name == name, "Unexpected source path")
        require(sha(blob("training/" + name)) == digest, "Source bytes differ: " + name)
    run_fields = {"selected_step", "history", "seconds", "model_sha256", "source_files_sha256"}
    require(set(training) - set(prior) == {"us_coverage"} and not set(prior) - set(training),
            "Unexpected recipe field changes")
    unchanged = sorted(set(prior) - run_fields)
    require(all(training[key] == prior[key] for key in unchanged), "Inherited recipe metadata changed")
    us = training["us_coverage"]
    require(us["seed"] == 167726 and us["positive_slots"] == list(range(8))
            and us["negative_slots"] == list(range(32, 44))
            and us["pack_sha256"] == "4a0d6febb8c0cd72b6bc58d26123d71b3cf3aa06736e013fd6ccae27dc58e6da",
            "US mixture recipe differs")
    require(us["source_sha256"] == training["source_files_sha256"]["us_coverage.py"],
            "US source hash differs")
    history = training["history"]
    require([item["step"] for item in history] == list(range(5000, 120001, 5000)),
            "Incomplete checkpoint history")
    best, chosen, eligible_points = None, None, 0
    for item in history:
        grid = item["grid"]
        require([row["threshold"] for row in grid] == THRESHOLDS, "Float grid changed")
        for row in grid:
            require(abs(row["macro_recall"] - sum(row["recall_by_engine"].values())
                        / len(row["recall_by_engine"])) < 1e-15, "Float macro differs")
            require(abs(row["ambient_faph"] - row["ambient_events"] / row["ambient_hours"]) < 1e-15,
                    "Float FAR arithmetic differs")
        eligible = [row for row in grid if row["ambient_faph"] <= .5 and row["max_source_false_rate"] <= .01]
        eligible_points += len(eligible)
        if eligible:
            selected = max(eligible, key=lambda row: (row["macro_recall"], -row["ambient_faph"], -row["max_source_false_rate"]))
            rank = (1, selected["macro_recall"], -selected["ambient_faph"], -selected["max_source_false_rate"])
        else:
            selected = min(grid, key=lambda row: (row["ambient_faph"], row["max_source_false_rate"], -row["macro_recall"]))
            rank = (0, -selected["ambient_faph"], -selected["max_source_false_rate"], selected["macro_recall"])
        require(selected == item["selected"], "Recorded checkpoint threshold differs")
        if best is None or rank > best:
            best, chosen = rank, item
    require(chosen["step"] == training["selected_step"] == export["selected_step"] == 35000,
            "Recorded checkpoint choice differs")
    calibration, background = reports["calibration"], reports["background"]
    require(calibration["synthetic_report_sha256"] == [sha(raw[key]) for key in ("phone", "vctk")]
            and calibration["background_report_sha256"] == sha(raw["background"]), "Gate input hashes differ")
    source_checks = [("calibration", "calibrate_phone_development.py"),
                     ("phone", "score_phone_development.py"), ("vctk", "score_phone_development.py"),
                     ("background", "evaluate_frozen_human.py"), ("export", "audit_stream_export.py")]
    for key, name in source_checks:
        require(reports[key]["source_sha256"] == sha(blob("training/" + name)), "Report source differs: " + key)
    trace_links, historical = {}, {}
    for key, count in (("phone", 80), ("vctk", 300)):
        report = reports[key]
        require(report["model_sha256"] == MODEL and len(report["records"]) == count
                and report["calibration_sha256"] == sha(raw["validation"])
                and report["threshold"] == reports["validation"]["threshold"] == .99609375,
                "Synthetic input identity differs")
        require(not report["final_data_used"] and not report["threshold_selected"], "Synthetic scope differs")
        positives = [item for item in report["records"] if item["record"]["target"] == "hey_nyx"]
        counts = {"positive_clips": len(positives), "positive_any_event_clips": sum(
            bool(item["nyx"]["event_seconds"]) for item in positives)}
        for target, group in ((None, "common_negative"), ("hi_esp", "cross-target_negative")):
            negatives = [item for item in report["records"] if item["record"]["target"] == target]
            counts[group] = dict(clips=len(negatives), false_events=sum(
                len(item["nyx"]["event_seconds"]) for item in negatives), false_activation_clips=sum(
                bool(item["nyx"]["event_seconds"]) for item in negatives))
        require(all(counts[field] == report["summaries"]["nyx"][field]
                    == reports["rejection"]["historical"][key][field] for field in counts),
                "Historical event accounting differs")
        historical[key] = counts
        for index, item in enumerate(report["records"]):
            stem = "phone" if key == "phone" else "vctk"
            trace_links[f".local/wakeword/v44-{stem}-development-historical-threshold-v1/{index}-scores.npy"] = item["nyx"]["score_sha256"]
    require(background["model_sha256"] == MODEL and len(background["speakers"]) == 87
            and background["frozen_report_sha256"] == sha(raw["validation"])
            and background["threshold"] == .99609375
            and background["sliding_window"] == calibration["sliding_window"] == 5
            and not background["independent_final_scored"], "Background identity/scope differs")
    for item in background["speakers"]:
        require(item["false_events"] == len(item["event_seconds"]), "Background event accounting differs")
        trace_links[".local/wakeword/v44-voxconverse-development-v1/" + item["speaker"] + "-scores.npy"] = item["score_sha256"]
    require(sum(item["false_events"] for item in background["speakers"]) == background["false_events"] == 0,
            "Background aggregate differs")
    require(abs(upper95(0, background["source_hours"]) - background["false_per_hour_upper_95"]) < 1e-12,
            "Historical background upper bound differs")
    receipt = reports["receipt"]
    require(receipt["authenticated_traces"] == len(receipt["traces"]) == len(trace_links) == 467,
            "Score receipt count differs")
    require({item["path"]: item["sha256"] for item in receipt["traces"]} == trace_links,
            "Score receipt report links differ")
    require(receipt["all_match"] and receipt["script_sha256"] == sha(blob("experiments/v44_score_receipt.py")),
            "Score receipt source differs")
    grid = calibration["grid"]
    require([row["threshold"] for row in grid] == sorted(set(THRESHOLDS + [.5 + i / 256 for i in range(128)])),
            "Fixed joint grid changed")
    max_upper_error = 0.
    for row in grid:
        require(set(row["groups"]) == {reports[key]["manifest_sha256"] for key in ("phone", "vctk")}
                and row["background_hours"] == background["source_hours"], "Joint input groups differ")
        groups = list(row["groups"].values())
        macro = sum(group["triggered"] / group["positives"] for group in groups) / len(groups)
        require(abs(macro - row["synthetic_macro_clip_trigger"]) < 1e-15, "Joint macro differs")
        upper = upper95(row["background_events"], row["background_hours"])
        max_upper_error = max(max_upper_error, abs(upper - row["background_upper_95"]))
        eligible = (upper <= .5 and macro >= .48125 and all(
            group["false_activation_clips"] == 0 and group["pre_generation_positive_events"] == 0 for group in groups))
        require(eligible == row["eligible"], "Joint eligibility differs")
    require(max_upper_error < 1e-12, "Poisson upper bounds differ")
    rejection = reports["rejection"]
    require(len(grid) == rejection["grid_points"] == 138 and sum(row["eligible"] for row in grid) == rejection["eligible_points"] == 0,
            "Rejection count differs")
    require(min(sum(group["false_activation_clips"] for group in row["groups"].values()) for row in grid)
            == rejection["minimum_negative_false_clips"] == 11, "Minimum false clips differs")
    require(max(row["synthetic_macro_clip_trigger"] for row in grid) == rejection["max_macro"], "Maximum macro differs")
    require(calibration["threshold"] is None and calibration["selected"] is None
            and not calibration["viable_operating_point"] and not calibration["final_data_used"], "Rejected gate changed")
    result = dict(scope="Independent source/report-link and recorded-count arithmetic audit; no score-array or model replay",
        gpu_commit=REF, model_sha256=MODEL, input_report_sha256={names[key]: sha(value) for key, value in raw.items()},
        source_files_authenticated=len(training["source_files_sha256"]), unchanged_v43_recipe_fields=unchanged,
        recorded_checkpoints_recomputed=len(history), float_grid_points_recomputed=sum(len(item["grid"]) for item in history),
        float_eligible_points=eligible_points, selected_step=chosen["step"], selected_float_row=chosen["selected"],
        joint_grid_points_recomputed=len(grid), joint_eligible_points=0, minimum_negative_false_clips=11,
        maximum_poisson_upper95_absolute_difference=max_upper_error, score_receipt_report_links_checked=len(trace_links),
        historical_saved_event_counts=historical,
        script_sha256=sha(Path(__file__).read_bytes()),
        limits="Saved counts/choices/hash links verified only. GPU receipt score-file claims not independently rehashed on Mac; no waveform, score array, inference, actual-float/INT8 replay, unused development, final, board, playback or microphone access. No recognition/FAR/latency/resource/call parity or promotion.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
