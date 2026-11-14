"""Relate authenticated persistent confusions to pinned training crop policy.

Source inspection and saved-event reduction only. No feature/model/audio reads.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "research/wakeword_p4/"
PIN = "e1f341a9"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository-root", type=Path, required=True)
    p.add_argument("--packet-root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    assert not a.report.exists()
    commit = subprocess.check_output(["git", "rev-parse", PIN], cwd=a.repository_root, text=True).strip()

    def public(name):
        return subprocess.check_output(["git", "show", commit + ":" + BASE + name], cwd=a.repository_root)

    lock_bytes = public("results/4090-hey-broad-paired-qat-formal-execution-v1.json")
    lock = json.loads(lock_bytes)
    names = ["training/train_stream.py", "experiments/audit_hey_broad_batches_v1.py",
             "experiments/schedule_hey_broad_v1.py", "experiments/run_hey_broad_paired_qat_v1.py"]
    sources = {name: public(name) for name in names}
    for name, data in sources.items():
        hashes = [v for k, v in lock["files"].items() if k.endswith("/" + name)]
        assert hashes and all(h == sha(data) for h in hashes), name
    tree = ast.parse(sources[names[0]])
    main_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    functions = {n.name: ast.unparse(n) for n in main_node.body if isinstance(n, ast.FunctionDef)}
    assert "row['keyword_frame'] + rng.integers(3, 28)" in functions["sample"]
    assert "row['keyword_frame']) - int(prefix_rng.integers(15, 36))" in functions["prefix_sample"]
    assert "sample(1, 32), sample(0, 48, mine=mine)" in functions["batch"]
    assert "[1.0] * 32 + [20.0] * 96" in functions["batch"]
    audit_tree = ast.parse(sources[names[1]])
    config = next(n.value for n in ast.walk(audit_tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "args" for t in n.targets))
    settings = {k.arg: ast.literal_eval(k.value) for k in config.keywords}
    assert settings["negative_candidates"] == 1 and settings["prefix_negatives"] == 16
    assert not settings["mask_prefix_loss"]
    assert settings["time_mask_width"] == settings["frequency_mask_width"] == settings["hard_ambient_count"] == 0
    runner = ast.unparse(ast.parse(sources[names[3]]))
    assert "env['batch'](mine=False, augment=False)" in runner
    assert "tf.reduce_mean(tf.reshape(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits), [-1]) * weight)" in runner
    results_dir = a.repository_root / BASE / "results"
    audit_bytes = (results_dir / "mac-fp32-reused-saved-array-independent-audit-v1.json").read_bytes()
    saved_audit = json.loads(audit_bytes)
    assert saved_audit["all_passed"] and saved_audit["points"] == 552
    expected = next(m for m in saved_audit["models"] if m["model"] == "hey_nyx-float_control")
    point_bytes = (a.packet_root / "diagnostic/hey_nyx-float_control.jsonl").read_bytes()
    assert sha(point_bytes) == expected["original_jsonl_sha256"]
    points = [json.loads(line) for line in point_bytes.splitlines()]
    assert len(points) == 138
    synthetic = [[r for r in p["records"] if r["phase"] == "synthetic"] for p in points]
    negative_sets = [{(r["source_id"], r["variant"]) for r in rows
                      if r["target_id"] not in ["hey_nyx", "hey_willow"] and r["events"]} for rows in synthetic]
    persistent = set.intersection(*negative_sets)
    assert len(persistent) == 13
    rows = []
    for key in sorted(persistent):
        records = [next(r for r in group if (r["source_id"], r["variant"]) == key) for group in synthetic]
        first_offsets = [r["events"][0]["generated_prefix_end_sample_signed_ms"] for r in records]
        after = sum(any(e["generated_prefix_end_sample_signed_ms"] >= 0 for e in r["events"]) for r in records)
        first_after = sum(offset >= 0 for offset in first_offsets)
        rows.append(dict(source_id=key[0], variant=key[1], original_points=len(points),
                         first_generated_prefix_offset_ms_range=[min(first_offsets), max(first_offsets)],
                         points_first_event_after_generated_prefix=first_after,
                         points_any_event_after_generated_prefix=after))
    positive_ends = list(range(3, 28))
    prefix_ends = sorted(-v for v in range(15, 36))
    gap = sorted(set(range(-35, 28)) - set(positive_ends) - set(prefix_ends))
    assert gap == list(range(-14, 3))
    assert all(r["points_first_event_after_generated_prefix"] == 138 for r in rows)
    result = dict(schema="mac-fp32-confusion-timing-and-sampler-source-v1", pinned_training_source_commit=commit,
                  original_training_execution_sha256=sha(lock_bytes), source_sha256={k: sha(v) for k, v in sources.items()},
                  original_saved_array_audit_sha256=sha(audit_bytes), original_nyx_float_point_jsonl_sha256=sha(point_bytes),
                  primary_model="hey_nyx-float_control", persistent_common_false_clips=13,
                  all_point_first_events_checked=13 * 138,
                  all_persistent_first_events_after_generated_prefix=True,
                  persistent_first_event_offset_ms_range=[min(r["first_generated_prefix_offset_ms_range"][0] for r in rows),
                                                          max(r["first_generated_prefix_offset_ms_range"][1] for r in rows)],
                  records=rows,
                  source_policy=dict(context_frames=225, positive_crop_end_delta_to_keyword_frame=positive_ends,
                                     explicit_prefix_negative_end_delta_to_keyword_frame=prefix_ends,
                                     gap_between_these_two_explicit_own_target_crop_stages=gap,
                                     slots=dict(positive=32, ordinary_negative_after_prefix_replacement=32,
                                                prefix_negative=16, ambient_negative=48),
                                     per_example_loss_weights=dict(positive=1, negative=20),
                                     per_batch_weight_mass=dict(positive=32, ordinary_negative=640, prefix_negative=320, ambient_negative=960),
                                     negative_to_positive_weight_mass_ratio=60,
                                     source_exponent=settings["source_exponent"], negative_candidates=1,
                                     mine=False, augment=False, mask_prefix_loss=False),
                  actual_effective_source_or_phrase_frequencies_measured=False,
                  model_invocations=0, media_reads=0, thresholds_selected=False, acceptance_scored=False,
                  auditor_sha256=sha(Path(__file__).read_bytes()),
                  limits="All persistent false first events occur after the generated allocation boundary, so these observed errors are not solely pre-boundary triggers. This is not a phonetic-end measurement or causal proof about a crop gap/loss weight. Source policy and earlier surviving-positive evidence do not measure actual per-phrase batch frequency. Integer-trained raw reconstructions are not their original quantized training execution. Reused single-voice diagnostic; no changes to labels, policy, models, gates or fresh data.")
    with a.report.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["persistent_common_false_clips", "all_point_first_events_checked",
                                           "all_persistent_first_events_after_generated_prefix", "persistent_first_event_offset_ms_range"]}))


if __name__ == "__main__":
    main()
