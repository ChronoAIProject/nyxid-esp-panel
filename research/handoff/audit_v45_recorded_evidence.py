"""Recompute pinned v45 recorded choices and joint rejection, without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
REF = "1ed3f65ea65e095aea7ad6222c8bf88a8b909a2d"
MODEL = "f5d175a0172d4fde70dce746fa358331c3fcdf590139c2e76113fe14c45e30af"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(relative):
    return subprocess.check_output(["git", "-C", str(HERE.parents[1]), "show",
        REF + ":research/wakeword_p4/" + relative])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output required")
    poisson_source = HERE / "audit_v44_recorded_evidence.py"
    old_audit = json.loads((HERE.parent / "wakeword_p4/results/v44-recorded-evidence-mac-audit-v1.json").read_bytes())
    require(sha(poisson_source.read_bytes()) == old_audit["script_sha256"], "Frozen Poisson implementation differs")
    previous_functions = runpy.run_path(str(poisson_source))
    upper95, thresholds = previous_functions["upper95"], previous_functions["THRESHOLDS"]
    names = {key: "4090-v45-" + tail + ".json" for key, tail in {
        "training": "training-v1", "export": "export-audit-v1", "validation": "validation-v1",
        "phone": "phone-development-v1", "vctk": "vctk-development-v1",
        "background": "voxconverse-development-v1", "calibration": "reused-development-exploratory-calibration-v1",
        "receipt": "score-receipt-v1", "summary": "old-stage-summary-v1"}.items()}
    names["prior"] = "4090-v44-training-v1.json"
    raw = {key: blob("results/" + name) for key, name in names.items()}
    reports = {key: json.loads(value) for key, value in raw.items()}
    training, prior, export = (reports[key] for key in ("training", "prior", "export"))
    require(sha(raw["training"]) == export["training_sha256"], "Training hash differs")
    require(training["model_sha256"] == export["model_sha256"] == MODEL, "Model identity differs")
    for name, digest in training["source_files_sha256"].items():
        require(Path(name).name == name and sha(blob("training/" + name)) == digest, "Training source differs: " + name)
    changed = {"selected_step", "history", "seconds", "model_sha256", "source_files_sha256", "us_coverage"}
    require(set(training) == set(prior) and all(training[key] == prior[key] for key in training if key not in changed),
            "Inherited recipe metadata differs")
    us = training["us_coverage"]
    require(all(us[key] == value for key, value in prior["us_coverage"].items() if key != "source_sha256"),
            "Original US source choices differ")
    require(us["source_sha256"] == training["source_files_sha256"]["us_coverage.py"]
            and us["temporal"]["seed"] == 177726 and us["temporal"]["positive_slots"] == list(range(8))
            and us["temporal"]["complete_negative_slots"] == list(range(32, 38))
            and us["temporal"]["timeline_negative_slots"] == list(range(38, 44)), "US temporal recipe differs")
    require(training["batchnorm_initial"] == training["batchnorm_selected"] and export["frozen_bn_unchanged"], "Frozen BN differs")
    history = training["history"]
    require([row["step"] for row in history] == list(range(5000, 120001, 5000)), "Incomplete history")
    best, chosen, float_eligible = None, None, 0
    for item in history:
        grid = item["grid"]
        require([row["threshold"] for row in grid] == thresholds, "Float grid differs")
        eligible = [row for row in grid if row["ambient_faph"] <= .5 and row["max_source_false_rate"] <= .01]
        float_eligible += len(eligible)
        if eligible:
            selected = max(eligible, key=lambda row: (row["macro_recall"], -row["ambient_faph"], -row["max_source_false_rate"]))
            rank = (1, selected["macro_recall"], -selected["ambient_faph"], -selected["max_source_false_rate"])
        else:
            selected = min(grid, key=lambda row: (row["ambient_faph"], row["max_source_false_rate"], -row["macro_recall"]))
            rank = (0, -selected["ambient_faph"], -selected["max_source_false_rate"], selected["macro_recall"])
        require(selected == item["selected"], "Recorded float choice differs")
        if best is None or rank > best:
            best, chosen = rank, item["step"]
    require(chosen == training["selected_step"] == export["selected_step"] == prior["selected_step"] == 35000,
            "v44/v45 selected-step comparison differs")
    calibration, background, summary = (reports[key] for key in ("calibration", "background", "summary"))
    require(calibration["synthetic_report_sha256"] == [sha(raw[key]) for key in ("phone", "vctk")]
            and calibration["background_report_sha256"] == sha(raw["background"]), "Joint input links differ")
    links, historical = {}, {}
    for key, count in (("phone", 80), ("vctk", 300)):
        report = reports[key]
        require(len(report["records"]) == count and report["model_sha256"] == MODEL
                and report["calibration_sha256"] == sha(raw["validation"])
                and report["threshold"] == reports["validation"]["threshold"] == .99609375, "Synthetic identity differs")
        rows = report["records"]
        counts = {"positive_any_event_clips": sum(bool(item["nyx"]["event_seconds"])
            for item in rows if item["record"]["target"] == "hey_nyx")}
        for target, group in ((None, "common_negative"), ("hi_esp", "cross-target_negative")):
            negatives = [item for item in rows if item["record"]["target"] == target]
            counts[group] = dict(clips=len(negatives), false_events=sum(len(item["nyx"]["event_seconds"]) for item in negatives),
                false_activation_clips=sum(bool(item["nyx"]["event_seconds"]) for item in negatives))
        require(all(counts[field] == report["summaries"]["nyx"][field] for field in counts), "Historical totals differ")
        historical[key] = counts
        for index, item in enumerate(rows):
            links[f".local/wakeword/v45-{key}-development-historical-threshold-v1/{index}-scores.npy"] = item["nyx"]["score_sha256"]
    require(background["model_sha256"] == MODEL and len(background["speakers"]) == 87
            and background["frozen_report_sha256"] == sha(raw["validation"]), "Background identity differs")
    for item in background["speakers"]:
        require(item["false_events"] == len(item["event_seconds"]), "Background event accounting differs")
        links[".local/wakeword/v45-voxconverse-development-v1/" + item["speaker"] + "-scores.npy"] = item["score_sha256"]
    require(sum(item["false_events"] for item in background["speakers"]) == background["false_events"] == 1,
            "Background aggregate differs")
    require(abs(upper95(1, background["source_hours"]) - background["false_per_hour_upper_95"]) < 1e-12,
            "Historical Poisson bound differs")
    receipt = reports["receipt"]
    require(len(links) == len(receipt["traces"]) == receipt["authenticated_traces"] == 467
            and {item["path"]: item["sha256"] for item in receipt["traces"]} == links, "Score receipt links differ")
    grid = calibration["grid"]
    require([row["threshold"] for row in grid] == sorted(set(thresholds + [.5 + i / 256 for i in range(128)])), "Joint grid differs")
    max_error = 0.
    for row in grid:
        groups = list(row["groups"].values())
        macro = sum(group["triggered"] / group["positives"] for group in groups) / len(groups)
        require(abs(macro - row["synthetic_macro_clip_trigger"]) < 1e-15, "Joint macro differs")
        upper = upper95(row["background_events"], row["background_hours"])
        max_error = max(max_error, abs(upper - row["background_upper_95"]))
        eligible = upper <= .5 and macro >= .48125 and all(
            group["false_activation_clips"] == 0 and group["pre_generation_positive_events"] == 0 for group in groups)
        require(eligible == row["eligible"], "Joint eligibility differs")
    require(max_error < 1e-12 and len(grid) == summary["fixed_grid_points"] == 138
            and sum(row["eligible"] for row in grid) == summary["eligible_points"] == 0, "Joint rejection differs")
    require(min(sum(group["false_activation_clips"] for group in row["groups"].values()) for row in grid)
            == summary["min_negative_false_clips"] == 34, "Minimum false clips differs")
    require(max(row["synthetic_macro_clip_trigger"] for row in grid) == summary["max_macro"] == .86875,
            "Maximum macro differs")
    require(calibration["threshold"] is None and not calibration["viable_operating_point"], "Rejected operating point differs")
    result = dict(scope="Independent recorded-report, source/hash-link and count arithmetic only; no inference or array replay",
        gpu_commit=REF, model_sha256=MODEL, input_report_sha256={names[key]: sha(value) for key, value in raw.items()},
        training_sources_authenticated=len(training["source_files_sha256"]), checkpoints_recomputed=len(history),
        float_grid_points_recomputed=sum(len(item["grid"]) for item in history), float_eligible_points=float_eligible,
        v44_selected_step=prior["selected_step"], v45_selected_step=chosen, joint_grid_points_recomputed=len(grid),
        joint_eligible_points=0, minimum_negative_false_clips=34, maximum_macro=.86875,
        maximum_poisson_upper95_absolute_difference=max_error, score_receipt_report_links_checked=len(links),
        historical_saved_event_counts=historical, historical_background_events=background["false_events"],
        historical_background_upper95=background["false_per_hour_upper_95"], script_sha256=sha(Path(__file__).read_bytes()),
        poisson_implementation_sha256=sha(poisson_source.read_bytes()),
        limits="Mac verifies source/report metadata, identities, saved event totals and arithmetic. GPU score-file authentication claims are not independently rehashed/replayed on Mac; no model/weights/score arrays, unused development/final, board or audio access. Both selected step numbers are35000; differing weights/time/calibration distributions remain, no unique causal or parity claim.")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
