"""Read original relocated GPU captures; independently audit every diagnostic point.

No inference, audio decoding, workpoint selection, or GPU evaluator imports.
The capture root preserves original relative .local/wakeword paths. Original
manifests, reports, NPZs, summary and JSONLs must retain their published bytes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np

BASE = "research/wakeword_p4/"
OUT = ".local/wakeword/fp32-reused-development-v1"
BOUNDARIES = ("generated_prefix_end_sample", "full_rir_prefix_end_sample",
              "generated_container_end_sample", "full_rir_container_end_sample")
POLICY = "fp32-to-binary64-fsum-five-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def means(raw):
    assert raw.dtype == np.float32 and raw.ndim == 1
    assert np.isfinite(raw).all() and np.all((raw >= 0) & (raw <= 1))
    values = raw.tolist()
    return np.asarray([math.fsum(values[i:i + 5]) / 5
                       for i in range(max(0, len(values) - 4))], dtype=np.float64)


def event_indices(averages, threshold):
    """Find alternating threshold/rearm crossings without a streaming FSM."""
    assert math.isfinite(threshold) and 0 < threshold <= 1
    high = np.flatnonzero(averages >= threshold)
    low = np.flatnonzero(averages < threshold / 2)
    result, seek = [], 0
    while True:
        h = int(np.searchsorted(high, seek))
        if h == len(high):
            return result
        index = int(high[h])
        result.append(index + 5)
        l = int(np.searchsorted(low, index + 67))
        if l == len(low):
            return result
        seek = int(low[l]) + 1


def poisson_cdf_at_upper_mean(count, mean):
    """Poisson lower tail for mean > count; no SciPy/gamma inversion shared."""
    assert type(count) is int and count >= 0 and mean > count
    term = total = 1.0
    for k in range(count, 0, -1):
        term *= k / mean
        total += term
        if term < total * 1e-17:
            break
    return math.exp(-mean + count * math.log(mean) - math.lgamma(count + 1)) * total


def poisson_upper95(count):
    lo, hi = float(count + 1), float(2 * (count + 1))
    while poisson_cdf_at_upper_mean(count, hi) > .05:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if poisson_cdf_at_upper_mean(count, mid) > .05:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def compare(expected, actual):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys()
        for key in expected:
            compare(expected[key], actual[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual)
        for a, b in zip(expected, actual, strict=True):
            compare(a, b)
    elif type(expected) is float:
        assert type(actual) in (float, int) and math.isclose(expected, actual, rel_tol=2e-9, abs_tol=1e-8), (expected, actual)
    else:
        assert type(expected) is type(actual) and expected == actual, (expected, actual)


def self_check():
    from float32_event_policy_mac_reference_v1 import event_steps
    rng = np.random.default_rng(20261114)
    traces = [np.zeros(512, np.float32), np.ones(512, np.float32),
              np.asarray([1] * 5 + [0] * 66 + [1] * 5 + [0] * 67 + [1] * 5, np.float32),
              np.asarray([1] * 5 + [0] * 10 + [1] * 100, np.float32),
              np.full(512, .501, np.float32),
              np.asarray([0, np.nextafter(np.float32(0), np.float32(1)), 1, .5] * 90, np.float32)]
    traces += [rng.random(512, dtype=np.float32) for _ in range(6)]
    grid = [i / 256 for i in range(128, 256)] + [.7, .8, .9, .95, .97, .98, .985, .99, .992, .994]
    thresholds = sorted(set(grid)) + [float(np.float32(.501)), 1.0]
    checks = 0
    for raw in traces + [np.zeros(n, np.float32) for n in range(5)]:
        avg = means(raw)
        for threshold in thresholds:
            assert event_indices(avg, threshold) == event_steps(raw, threshold)
            checks += 1
    quantile_checks = 0
    for n in range(1, 101):
        values = rng.normal(size=n).tolist()
        for fraction in [.5, .95]:
            assert math.isclose(percentile(values, fraction), float(np.percentile(values, fraction * 100)), abs_tol=1e-12)
            quantile_checks += 1
    anchors = {0: -math.log(.05), 1: 4.743864518390577, 2: 6.295793621871988,
               10: 16.962219235721903}
    for count, expected in anchors.items():
        assert math.isclose(poisson_upper95(count), expected, rel_tol=1e-12)
    for count in [100, 1000, 10000, 20000]:
        assert math.isclose(poisson_cdf_at_upper_mean(count, poisson_upper95(count)), .05, abs_tol=1e-10)
    return dict(schema="mac-fp32-reused-saved-array-auditor-constructed-v1",
                event_trace_comparisons=checks, independent_percentile_comparisons=quantile_checks,
                poisson_analytic_or_tabulated_anchors=len(anchors), large_count_tail_checks=4,
                all_passed=True, model_invocations=0, media_reads=0, actual_probability_reads=0,
                scorer_sha256=sha(Path(__file__).read_bytes()),
                reference_sha256=sha(Path(__file__).with_name("float32_event_policy_mac_reference_v1.py").read_bytes()),
                limits="Constructed oracle checks only; not an actual capture, recognition, FAR or P4 acceptance result.")


def audit(repo, commit, root):
    def public(name):
        return subprocess.check_output(["git", "show", commit + ":" + BASE + name], cwd=repo)

    def read(original, expected_hash=None):
        path = PurePosixPath(original)
        assert not path.is_absolute() and ".." not in path.parts and original == path.as_posix()
        file = root / original
        assert file.resolve().is_relative_to(root.resolve())
        data = file.read_bytes()
        if expected_hash is not None:
            assert sha(data) == expected_hash, original
        return data

    lock_bytes = public("results/4090-fp32-reused-development-execution-v1.json")
    lock = json.loads(lock_bytes)
    score_lock_bytes = public("results/4090-fp32-reused-scoring-execution-v1.json")
    score_lock = json.loads(score_lock_bytes)
    collection_bytes = public("results/4090-fp32-reused-collection-original-result-v1.json")
    original = json.loads(collection_bytes)
    collection_audit_bytes = public("results/4090-fp32-reused-collection-independent-audit-v1.json")
    collection_audit = json.loads(collection_audit_bytes)
    assert original["execution_sha256"] == sha(lock_bytes) and original["all_passed"]
    assert collection_audit["execution_sha256"] == sha(lock_bytes) and collection_audit["all_passed"]
    assert score_lock["files"][BASE + "results/4090-fp32-reused-collection-independent-audit-v1.json"] == sha(collection_audit_bytes)
    assert read(OUT + "/result.json") == collection_bytes
    assert not lock["threshold_selected"] and not lock["acceptance_scored"] and lock["retries"] == 0
    assert score_lock["grid"] == lock["grid"] and len(lock["grid"]) == 138
    assert score_lock["versions"]["scipy"] and score_lock["threshold_selected"] is False
    assert score_lock["files"][BASE + "results/4090-fp32-reused-development-execution-v1.json"] == sha(lock_bytes)
    prior = json.loads((repo / (BASE + "results/mac-fp32-internal-trained-intake-v1.json")).read_text())
    assert {m["target"] + "-" + m["condition"]: m["sha256"] for m in lock["models"]} == {
        k: v["model_sha256"] for k, v in prior["contracts"].items()}
    manifests = {}
    for phase, material in lock["materials"].items():
        assert material["role"] == "reused-development"
        rows = json.loads(read(material["path"], material["sha256"]))["records"]
        assert len(rows) == material["records"] == {"synthetic": 360, "background": 12}[phase]
        manifests[phase] = rows
    original_order = [(phase, i, row) for phase, rows in manifests.items() for i, row in enumerate(rows)]
    assert len(original_order) == 372
    background_samples = sum(row["samples"] for row in manifests["background"])
    exposure = background_samples / 57600000
    audit_receipt_bytes = public("results/4090-fp32-reused-scoring-independent-audit-v1.json")
    audit_receipt = json.loads(audit_receipt_bytes)
    assert audit_receipt["all_passed"] and audit_receipt["points"] == 552
    summary = json.loads(read(OUT + "/diagnostic/summary.json", audit_receipt["summary_sha256"]))
    assert summary["execution_sha256"] == sha(lock_bytes) and len(summary["points"]) == 552
    assert all(summary[k] is False for k in ["threshold_selected", "acceptance_scored", "candidate_promoted"])
    shared, checked_models, summaries, compared_events, far_error = {}, [], [], 0, 0.0
    for model in lock["models"]:
        name = model["target"] + "-" + model["condition"]
        folder = OUT + "/" + name
        model_result = next(m for m in original["models"] if m["model"] == name)
        report_bytes = read(folder + "/report.json", model_result["report_sha256"])
        report = json.loads(report_bytes)
        collection_model_audit = next(m for m in collection_audit["models"] if m["model"] == name)
        records_bytes = read(folder + "/records.jsonl", collection_model_audit["records_sha256"])
        records = [json.loads(line) for line in records_bytes.splitlines()]
        assert records == report["records"] and report["model"] == model and report["all_passed"]
        assert report["execution_sha256"] == sha(lock_bytes) and len(records) == 372
        averages, steps_total = [], 0
        for rec, (phase, index, row) in zip(records, original_order, strict=True):
            assert rec["record"] == row and (rec["model"], rec["phase"], rec["index"]) == (name, phase, index)
            trace_path = folder + f"/{phase}/{index:06}.npz"
            assert rec["trace_path"] == trace_path and rec["repeat_exact"]
            assert score_lock["files"][trace_path] == rec["trace_sha256"]
            data = read(trace_path, rec["trace_sha256"])
            from io import BytesIO
            with np.load(BytesIO(data), allow_pickle=False) as z:
                assert set(z.files) == {"probability", "probability_repeat", "sample_end", "support_unclamped"}
                n = (max(0, (row["samples"] + 159) // 160 - 2) + 2) // 3
                assert n == rec["steps"]
                for key in ["probability", "probability_repeat"]:
                    assert z[key].shape == (n,) and z[key].dtype == np.float32
                    assert np.isfinite(z[key]).all() and np.all((z[key] >= 0) & (z[key] <= 1))
                assert z["probability"].tobytes() == z["probability_repeat"].tobytes()
                support = (3 * np.arange(1, n + 1, dtype=np.int64) + 2) * 160
                for key, expected in [("support_unclamped", support), ("sample_end", np.minimum(support, row["samples"]))]:
                    assert z[key].dtype == np.int64 and z[key].shape == (n,) and np.array_equal(z[key], expected)
                averages.append(means(z["probability"]))
            identity = (rec["input_sha256"], rec["raw_frontend_sha256"], row["pcm_sha256"], row["samples"], n)
            key = (phase, index)
            if key in shared:
                assert shared[key] == identity
            else:
                shared[key] = identity
            steps_total += n
        receipt = next(m for m in audit_receipt["models"] if m["model"] == name)
        points_bytes = read(OUT + "/diagnostic/" + name + ".jsonl", receipt["records_sha256"])
        points = [json.loads(line) for line in points_bytes.splitlines()]
        assert len(points) == 138 and [p["summary"]["threshold"] for p in points] == lock["grid"]
        for point in points:
            s = point["summary"]
            assert s["model"] == name and s["event_arithmetic"] == POLICY and s["threshold_selected"] is False
            assert len(point["records"]) == 372
            positives, negatives, bg, pre = [], {k: dict(clips=0, activated_clips=0, events=0) for k in ["common", "cross_target"]}, 0, 0
            for rec, avg, observed in zip(records, averages, point["records"], strict=True):
                row = rec["record"]
                steps = event_indices(avg, s["threshold"])
                times = [min((3 * step + 2) * 160, row["samples"]) for step in steps]
                expected = dict(phase=rec["phase"], index=rec["index"], source_id=row["source_id"],
                                events=[dict(step=step, sample=t) for step, t in zip(steps, times, strict=True)])
                compared_events += len(steps)
                if rec["phase"] == "background":
                    bg += len(steps)
                else:
                    expected.update({k: row[k] for k in ["target_id", "variant", "carrier", "length_scale"]})
                    for event in expected["events"]:
                        event.update({b + "_signed_ms": (event["sample"] - row[b]) / 16 for b in BOUNDARIES})
                    if row["target_id"] == model["target"]:
                        after = [t for t in times if t >= row["lead_samples"]]
                        complete = any(t >= row["generated_prefix_end_sample"] for t in after)
                        pre += sum(t < row["lead_samples"] for t in times)
                        hit = dict(any_after_source=bool(after), complete=complete, before_only=bool(after) and not complete,
                                   missing=not after, latency_ms=(after[0] - row["generated_prefix_end_sample"]) / 16 if after else None,
                                   carrier=row["carrier"], variant=row["variant"])
                        positives.append(hit)
                        expected["positive"] = hit
                    else:
                        other = "hey_nyx" if model["target"] == "hey_willow" else "hey_willow"
                        neg = negatives["cross_target" if row["target_id"] == other else "common"]
                        neg["clips"] += 1
                        neg["activated_clips"] += int(bool(steps))
                        neg["events"] += len(steps)
                assert observed == expected
            assert len(positives) == 36 and negatives["common"]["clips"] == 288 and negatives["cross_target"]["clips"] == 36
            groups = [("all", positives), ("standalone", [p for p in positives if p["carrier"] == ""]),
                      ("command", [p for p in positives if p["carrier"] != ""])]
            groups += [(f"variant{v}", [p for p in positives if p["variant"] == v]) for v in sorted({p["variant"] for p in positives})]
            positive_summary = {}
            for key, group in groups:
                latency = [p["latency_ms"] for p in group if p["latency_ms"] is not None]
                positive_summary[key] = dict(clips=len(group), **{k: sum(p[k] for p in group) for k in ["any_after_source", "complete", "before_only", "missing"]},
                                            signed_latency_ms=dict(median=percentile(latency, .5), p95=percentile(latency, .95), observed=len(latency)))
            bound = poisson_upper95(bg) / exposure
            far_error = max(far_error, abs(bound - s["background"]["far_upper95"]))
            expected_summary = dict(model=name, threshold=s["threshold"], event_arithmetic=POLICY,
                                    positive=positive_summary, negative=negatives, pre_source_events=pre,
                                    background=dict(events=bg, hours=exposure, far_upper95=bound), threshold_selected=False)
            compare(expected_summary, s)
            summaries.append(s)
        checked_models.append(dict(model=name, original_report_sha256=sha(report_bytes), original_jsonl_sha256=sha(points_bytes),
                                   complete_inputs=372, probability_values_per_repeat=steps_total, diagnostic_points=138))
        print(json.dumps(checked_models[-1]), flush=True)
    assert summaries == summary["points"] and len(shared) == 372
    return dict(schema="mac-fp32-reused-saved-array-independent-audit-v1", gpu_commit=commit,
                execution_sha256=sha(lock_bytes), score_execution_sha256=sha(score_lock_bytes),
                gpu_score_audit_sha256=sha(audit_receipt_bytes), models=checked_models,
                points=552, input_point_records=552 * 372, events_checked=compared_events,
                background_samples=background_samples, background_hours=exposure,
                maximum_independent_far_absolute_difference=far_error, all_passed=True,
                model_invocations=0, media_reads=0, thresholds_selected=False, acceptance_scored=False,
                auditor_sha256=sha(Path(__file__).read_bytes()),
                limits="Authenticated original saved arrays, source manifests, reports, trace clocks/repeats and all emitted diagnostic events/metrics. FAR independently inverted from Poisson tail without SciPy. Shared frontend/input identities are authenticated receipt hashes, not recomputed PCM/features. Reused development only; no operating point selection, recognition qualification or P4 measurement. Generated boundaries are not measured phonetic ends.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--repository-root", type=Path)
    p.add_argument("--gpu-commit")
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    if a.self_check:
        assert not any([a.repository_root, a.gpu_commit, a.capture_root])
        result = self_check()
    else:
        assert all([a.repository_root, a.gpu_commit, a.capture_root])
        result = audit(a.repository_root, a.gpu_commit, a.capture_root)
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
