"""Summarize verified Hi ESP PCM results, optionally against a frozen Hey Nyx.

Thresholds come from a separate validation calibration, never these fixtures.
Only aggregates are exported; private WAV paths and per-clip scores stay local.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys


def validate_engine(report):
    if report.get("engine") != "wn9_hiesp":
        raise ValueError("This historical Hi ESP summarizer does not support other reference engines")


def summarize(rows, target):
    positives = [r for r in rows if r["target"] == target]
    ordinary = [r for r in rows if r["target"] == "none"]
    esp_near_texts = {"Hi SP.", "Hi ESB.", "Hi ASP.", "Hi EST.", "I see ESP.", "High speed.",
                      "Hi S P.", "Hi E S B.", "Hi A S P.", "Hi E S T.", "I see E S P."}
    other = [r for r in rows if r["target"] not in (target, "none")]
    nyx_near_texts = {"Hey Nick.", "Hey Next.", "Hey Mix.", "Hey Six.", "Hey Max.",
                      "Hey Nice.", "Hey Nexus.", "Hey Alex.", "Hey Siri.",
                      "Hey Nick, turn on the light.", "Hey Next, what time is it?"}
    latency = []
    timed_hits = 0
    for row in positives:
        end = row.get("keyword_end")
        if end is None:
            continue
        delays = [(at-end)/16 for at in row["result"]["offsets"]]
        valid = [ms for ms in delays if -100 <= ms <= 800]
        if valid:
            timed_hits += 1
            latency.append(valid[0])

    def count(group):
        return {"clips": len(group), "clips_with_event": sum(r["result"]["events"] > 0 for r in group),
                "events": sum(r["result"]["events"] for r in group)}

    return {"positives_any_event": count(positives),
            "positives_in_estimated_end_window": {"clips": sum(r.get("keyword_end") is not None for r in positives),
                "detected": timed_hits, "window_ms": [-100, 800]},
            "ordinary_and_near_word_negatives": count(ordinary),
            "shared_negative_texts": count([r for r in ordinary if r["text"] not in esp_near_texts]),
            "shared_ordinary_sentences": count([r for r in ordinary if r["text"] not in (esp_near_texts | nyx_near_texts)]),
            "nyx_specific_near_words": count([r for r in ordinary if r["text"] in nyx_near_texts]),
            "negative_events_by_text": {text: count([r for r in ordinary if r["text"] == text])
                                        for text in sorted({r["text"] for r in ordinary})},
            "hi_esp_specific_near_words_unvalidated_pronunciation": count(
                [r for r in ordinary if r["text"] in esp_near_texts]),
            "other_wake_phrase": count(other),
            "latency_ms_from_estimated_end_median": statistics.median(latency) if latency else None,
            "latency_ms_from_estimated_end_range": [min(latency), max(latency)] if latency else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if bool(args.model) != bool(args.calibration):
        parser.error("Both --model and --calibration are required for comparison")
    if args.output.exists():
        parser.error("Do not overwrite an existing result")
    report = json.loads(args.baseline.read_text())
    validate_engine(report)
    fixtures = list(map(json.loads, args.manifest.read_text().splitlines()))
    if not report.get("restored") or len(report["clips"]) != len(fixtures):
        raise ValueError("Benchmark incomplete or original development app not verified restored")
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != report["manifest_sha256"]:
        raise ValueError("Manifest does not match benchmark")
    for fixture, row in zip(fixtures, report["clips"]):
        if any(fixture[k] != row[k] for k in ("path", "sha256", "samples")) or not row["result"]["complete"]:
            raise ValueError("Fixture/result identity mismatch")
    result = {"scope": "Paired Mac synthetic development regression, not acoustic acceptance.",
              "manifest_sha256": report["manifest_sha256"], "probe_sha256": report["probe_sha256"],
              "clips": len(fixtures), "all_transfers_crc_verified": True,
              "original_app_restoration_verified": True,
              "hiesp_thresholds": sorted({r["threshold"] for r in report["clips"]}),
              "fixtures_by_scope": dict(Counter(r.get("scope", "paired_mac_regression") for r in fixtures)),
              "limits": ["Synthetic variants of one source phrase are correlated; these are not independent human trials.",
                         "Keyword endpoints are RMS estimates, not phonetic alignment. Report any-event and fixed-window recall separately.",
                         "The phrase-specific negative texts differ; do not equate the two near-word distributions.",
                         "Mac pronunciation diagnostics are imperfect. Older bare ESP samples were rejected for abbreviation expansion.",
                         "No long continuous-negative, real-user, microphone/AEC or full-app load acceptance is established here."]}

    def groups(rows, target):
        paired = [r for r in rows if r.get("scope", "paired_mac_regression") == "paired_mac_regression"]
        controls = [r for r in rows if r.get("scope") == "previous_acoustic_fixture_control"]
        return {"paired": summarize(paired, target), "controls": summarize(controls, target),
                "by_voice_and_variant": {f"{voice}/variant-{variant}": summarize(
                    [r for r in paired if r["voice"] == voice and r["variant"] == variant], target)
                    for voice, variant in sorted({(r["voice"], r["variant"]) for r in paired})}}

    result["hi_esp"] = groups(report["clips"], "hi_esp")
    calls = sum(r["result"]["calls"] for r in report["clips"])
    result["hiesp_standalone_inference"] = {
        "calls": calls, "chunk_samples": 512,
        "mean_us_per_chunk": sum(r["result"]["total_us"] for r in report["clips"])/calls,
        "max_us_per_chunk": max(r["result"]["max_us"] for r in report["clips"])}
    if args.model:
        import numpy as np
        import soundfile as sf
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"wakeword_p4"/"training"))
        from evaluate import Detector, smooth
        from detector_policy import calibrated_smoothing_steps
        from features import features
        calibration = json.loads(args.calibration.read_text())
        smoothing_steps = calibrated_smoothing_steps(calibration)
        if calibration.get("model_sha256") != hashlib.sha256(args.model.read_bytes()).hexdigest():
            raise ValueError("Calibration does not identify this frozen model; recalibrate in a fresh directory")
        selected = calibration["selected"]
        threshold = selected["threshold"]
        detector = Detector(args.model)
        rows = []
        for number, fixture in enumerate(fixtures):
            path = args.manifest.parent/fixture["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != fixture["sha256"]:
                raise ValueError("Fixture WAV hash mismatch")
            pcm, rate = sf.read(path, dtype="int16")
            if rate != 16000 or pcm.ndim != 1 or len(pcm) != fixture["samples"]:
                raise ValueError("Expected exact mono 16 kHz PCM fixture")
            detector.reset()
            values = smooth(detector.predict(features(pcm)), smoothing_steps)
            offsets = []
            armed, last = True, -1000
            for index, value in enumerate(values):
                if value < threshold*.5 and index-last >= 67:
                    armed = True
                if armed and value >= threshold and index-last >= 67:
                    offsets.append(int((3*(index+smoothing_steps-1)+5)*160))
                    last, armed = index, False
            rows.append({**fixture, "result": {"events": len(offsets), "offsets": offsets}})
            if number % 50 == 0:
                print(json.dumps({"phase": "frozen_hey_nyx", "completed": number+1, "total": len(fixtures)}), flush=True)
        result["hey_nyx"] = groups(rows, "hey_nyx")
        result["hey_nyx_model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
        result["hey_nyx_calibration_sha256"] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
        result["hey_nyx_validation_selected_threshold"] = threshold
        result["hey_nyx_smoothing_steps"] = smoothing_steps
        result["hey_nyx_calibration_scope"] = calibration.get("selection_scope")
        result["hey_nyx_validation_operating_point"] = calibration.get("viable_operating_point")
        if not calibration.get("viable_operating_point"):
            result["limits"].append("This calibration does not pass the original validation operating constraints. Paired results are diagnostic, not candidate acceptance.")
        result["limits"].append("Hey Nyx here uses host BUILTIN_REF INT8 inference; P4 numeric and timing verification is separate.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("clips", "hi_esp", "hey_nyx") if k in result}, indent=2))


if __name__ == "__main__":
    main()
