"""Fixed-grid diagnostic of already-saved training witnesses; no inference."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_fp32_reused_saved_arrays_mac_v1 import means, event_indices

BASE = Path("research/wakeword_p4")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def signed_ms(sample, boundary):
    return (sample - boundary) / 16 if boundary is not None else None


def create_plan(repo, training_packet, development_packet, plan_path):
    intake = repo / BASE / "results/mac-fp32-internal-fullstream-intake-v1.json"
    report = json.loads(intake.read_bytes())
    assert report["all_passed"] and len(report["cases"]) == 540
    training_manifest = training_packet / "manifest.json"
    assert file_sha(training_manifest) == report["manifest_sha256"]
    requests_path = training_packet / "receipts/4090-fp32-internal-fullstream-execution-v1.json"
    assert file_sha(requests_path) == json.loads(training_manifest.read_bytes())["files"][str(requests_path.relative_to(training_packet))]["sha256"]
    request_lock = json.loads(requests_path.read_bytes())
    grid_path = development_packet / "receipts/4090-fp32-reused-development-execution-v1.json"
    old_audit = repo / BASE / "results/mac-fp32-reused-saved-array-independent-audit-v1.json"
    assert file_sha(grid_path) == json.loads(old_audit.read_bytes())["execution_sha256"]
    grid_lock = json.loads(grid_path.read_bytes())
    assert len(request_lock["requests"]) == 135 and len(grid_lock["grid"]) == 138
    assert {m["target"] + "-" + m["condition"]: m["sha256"] for m in grid_lock["models"]} == {
        k: v["model_sha256"] for k, v in report["contracts"].items()}
    source = Path(__file__).resolve()
    dependencies = [source, source.with_name("audit_fp32_reused_saved_arrays_mac_v1.py")]
    failure = repo / BASE / "results/mac-original-training-witness-events-initial-setup-failure-v1.json"
    boundary_failure = repo / BASE / "results/mac-original-training-witness-events-missing-boundary-reader-failure-v2.json"
    locked = {str(p.relative_to(repo)): file_sha(p) for p in dependencies + [intake, old_audit, failure, boundary_failure]}
    packet_files = {"training_requests": dict(member=str(requests_path.relative_to(training_packet)), sha256=file_sha(requests_path)),
                    "development_grid": dict(member=str(grid_path.relative_to(development_packet)), sha256=file_sha(grid_path))}
    plan = dict(schema="mac-original-training-witness-events-plan-v1", files=locked, packet_files=packet_files,
                original_training_archive_sha256=report["archive_sha256"], models=grid_lock["models"],
                fixed_grid=grid_lock["grid"], complete_requests=135,
                request_orders=[r["order"] for r in request_lock["requests"]],
                index_contract="Packet position indexes arrays; retain distinct original request.order without renumbering",
                prefix_annotation_policy="Preserve absent negative prefix annotations as null; never infer from container end",
                prefix_annotation_counts=dict(present=sum(r["generated_prefix_end_sample"] is not None for r in request_lock["requests"]),
                                              absent=sum(r["generated_prefix_end_sample"] is None for r in request_lock["requests"])),
                source_role="already-consumed-training-witness",
                own_positive_derived_clips=27, reciprocal_positive_clips=27, common_negative_clips=81,
                backends=["tf", "litert"], state_policy="Original complete cold-start streams, no crop/reset/removal",
                event_policy="Original raw float32 -> binary64 fsum5, 67-step cooldown and half-threshold rearm",
                event_clock="min((3*one_based_step+2)*160,original_samples)",
                groups=["all-own", "own-engine", "own-variant", "common-phrase", "common-engine", "reciprocal"],
                probability_invocations=0, thresholds_selected=False, acceptance_scored=False,
                limits="Diagnostic of original already-consumed training examples only, not generalization. No independent background/FAR exists here. Integer-trained raw reconstructions are not their original quantized training execution. All138 points and original metadata retained; no candidate promotion, new model export, feature generation, audio, fresh development or acceptance.")
    with plan_path.open("x") as f:
        f.write(json.dumps(plan, indent=2) + "\n")
    return plan


def count_range(values):
    return [min(values), max(values)]


def run(repo, training_packet, development_packet, plan_path, output, report_path):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    for name, digest in plan["files"].items():
        assert file_sha(repo / name) == digest, name
    for key, root in [("training_requests", training_packet), ("development_grid", development_packet)]:
        assert file_sha(root / plan["packet_files"][key]["member"]) == plan["packet_files"][key]["sha256"]
    prior = json.loads((repo / BASE / "results/mac-fp32-internal-fullstream-intake-v1.json").read_bytes())
    request_lock = json.loads((training_packet / plan["packet_files"]["training_requests"]["member"]).read_bytes())
    requests = request_lock["requests"]
    assert len(requests) == len(set(r["order"] for r in requests)) == 135
    assert [r["order"] for r in requests] == plan["request_orders"]
    for request in requests:
        assert all(type(request[k]) is int for k in ["samples", "lead_samples", "generated_container_end_sample"])
        assert type(request["generated_prefix_end_sample"]) is int if request["source"]["kind"] == "positive" else request["generated_prefix_end_sample"] is None
    assert plan["prefix_annotation_counts"] == dict(present=54, absent=81)
    output.mkdir(exist_ok=False)
    ledger = output / "original-training-witness-events.jsonl.gz"
    summaries, differences, events_total = [], 0, 0
    model_descriptions = []
    with gzip.open(ledger, "xt", encoding="utf-8") as log:
        for model in plan["models"]:
            name = model["target"] + "-" + model["condition"]
            captured = []
            for index, request in enumerate(requests):
                member = "arrays/" + name + f"-{index:03}.npz"
                path = training_packet / member
                case = next(c for c in prior["cases"] if c["model"] == name and c["index"] == index)
                assert case["request_order"] == request["order"]
                assert file_sha(path) == case["original_npz_sha256"]
                with np.load(path, allow_pickle=False) as z:
                    support = (3 * np.arange(1, case["steps"] + 1, dtype=np.int64) + 2) * 160
                    assert np.array_equal(z["support_unclamped"], support)
                    assert np.array_equal(z["sample_end"], np.minimum(support, request["samples"]))
                    backend_means = {}
                    for backend in plan["backends"]:
                        assert z[backend].shape == (case["steps"],) and z[backend].dtype == np.float32
                        assert z[backend].tobytes() == z[backend + "_repeat"].tobytes()
                        backend_means[backend] = means(z[backend])
                captured.append(backend_means)
            own_rows = [r for r in requests if r["source"]["target"] == model["target"]]
            assert len(own_rows) == 27
            engine_denominators = Counter(r["source"]["engine"] for r in own_rows)
            assert dict(engine_denominators) == dict(piper=12, kokoro=12, qwen=3)
            model_points = []
            for threshold in plan["fixed_grid"]:
                records = {backend: [] for backend in plan["backends"]}
                for index, (request, avg) in enumerate(zip(requests, captured, strict=True)):
                    source = request["source"]
                    role = "own" if source["target"] == model["target"] else "common" if source["kind"] == "negative" else "reciprocal"
                    per_backend_steps = []
                    for backend in plan["backends"]:
                        steps = event_indices(avg[backend], threshold)
                        times = [min((3 * s + 2) * 160, request["samples"]) for s in steps]
                        after = [t for t in times if t >= request["lead_samples"]]
                        records[backend].append(dict(index=index, request_order=request["order"], source_id=source["id"], engine=source["engine"],
                                                     source_text=source["text"], role=role, variant=request["variant"],
                                                     prefix_annotation_available=request["generated_prefix_end_sample"] is not None,
                                                     events=[dict(step=s, sample=t, prefix_signed_ms=signed_ms(t, request["generated_prefix_end_sample"]),
                                                                  container_signed_ms=signed_ms(t, request["generated_container_end_sample"]))
                                                             for s, t in zip(steps, times, strict=True)],
                                                     any_after_source=bool(after),
                                                     complete_generated_prefix=any(t >= request["generated_prefix_end_sample"] for t in after) if request["generated_prefix_end_sample"] is not None else None,
                                                     pre_source_events=sum(t < request["lead_samples"] for t in times),
                                                     first_after_source_prefix_signed_ms=signed_ms(after[0], request["generated_prefix_end_sample"]) if after else None,
                                                     first_after_source_container_signed_ms=signed_ms(after[0], request["generated_container_end_sample"]) if after else None))
                        events_total += len(steps)
                        per_backend_steps.append(steps)
                    differences += per_backend_steps[0] != per_backend_steps[1]
                for backend in plan["backends"]:
                    group = records[backend]
                    own = [r for r in group if r["role"] == "own"]
                    common = [r for r in group if r["role"] == "common"]
                    reciprocal = [r for r in group if r["role"] == "reciprocal"]
                    assert (len(own), len(common), len(reciprocal)) == (27, 81, 27)
                    engines = {e: dict(clips=engine_denominators[e],
                                       any_after_source=sum(r["any_after_source"] for r in own if r["engine"] == e),
                                       complete=sum(r["complete_generated_prefix"] for r in own if r["engine"] == e))
                               for e in engine_denominators}
                    common_phrases = {text: dict(clips=sum(r["source_text"] == text for r in common),
                                                activated_clips=sum(bool(r["events"]) for r in common if r["source_text"] == text))
                                      for text in sorted({r["source_text"] for r in common})}
                    summary = dict(model=name, backend=backend, threshold=threshold, own_clips=27,
                                   any_after_source=sum(r["any_after_source"] for r in own),
                                   complete=sum(r["complete_generated_prefix"] for r in own),
                                   missing=sum(not r["any_after_source"] for r in own), own_engine=engines,
                                   common_clips=81, common_activated_clips=sum(bool(r["events"]) for r in common),
                                   common_phrase=common_phrases, reciprocal_clips=27,
                                   reciprocal_activated_clips=sum(bool(r["events"]) for r in reciprocal),
                                   pre_source_events=sum(r["pre_source_events"] for r in group), threshold_selected=False)
                    log.write(json.dumps(dict(summary=summary, records=group)) + "\n")
                    summaries.append(summary)
                    model_points.append(summary)
            descriptions = {}
            for backend in plan["backends"]:
                rows = [s for s in model_points if s["backend"] == backend]
                descriptions[backend] = dict(registered_points=len(rows),
                    ranges={k: count_range([s[k] for s in rows]) for k in ["any_after_source", "complete", "missing", "common_activated_clips", "reciprocal_activated_clips"]},
                    own_engine={e: dict(clips=engine_denominators[e],
                                        any_after_source_range=count_range([s["own_engine"][e]["any_after_source"] for s in rows]),
                                        complete_range=count_range([s["own_engine"][e]["complete"] for s in rows])) for e in engine_denominators},
                    common_phrase={text: dict(clips=rows[0]["common_phrase"][text]["clips"],
                                              activated_clip_range=count_range([s["common_phrase"][text]["activated_clips"] for s in rows]))
                                   for text in rows[0]["common_phrase"]})
            model_descriptions.append(dict(model=name, derived_own_clips=27, unique_own_source_utterance_ids=len({r["source"]["id"] for r in own_rows}),
                                           backends=descriptions))
            print(json.dumps(model_descriptions[-1]), flush=True)
    assert len(summaries) == 1104
    result = dict(schema="mac-original-training-witness-events-v1", plan_sha256=sha(plan_bytes),
                  original_training_archive_sha256=plan["original_training_archive_sha256"],
                  models=model_descriptions, points_including_both_backends=1104, case_points=1104*135,
                  emitted_events=events_total, tf_litert_event_sequence_differences=differences,
                  ledger_sha256=file_sha(ledger), ledger_bytes=ledger.stat().st_size,
                  model_invocations=0, audio_decodes=0, thresholds_selected=False, acceptance_scored=False,
                  source_sha256=file_sha(Path(__file__)), limits=plan["limits"])
    with report_path.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["points_including_both_backends", "case_points", "tf_litert_event_sequence_differences", "ledger_bytes"]}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "run"])
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--training-packet", type=Path, required=True)
    p.add_argument("--development-packet", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    if a.action == "plan":
        create_plan(a.repository_root, a.training_packet, a.development_packet, a.plan)
    else:
        assert a.output_dir is not None and a.report is not None and not a.report.exists()
        run(a.repository_root, a.training_packet, a.development_packet, a.plan, a.output_dir, a.report)


if __name__ == "__main__":
    main()
