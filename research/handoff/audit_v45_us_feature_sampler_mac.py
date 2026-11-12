"""Replay the frozen training-only feature/sampler plan on original Linux bytes.

Uses the isolated NumPy environment already recorded by the Mac source audit.
No model, waveform, device, playback or microphone APIs are imported.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PLAN = HERE / "plans/v45-us-feature-sampler-mac-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output directory required")
    require(sys.flags.optimize == 0, "Do not disable the original helper's assertions")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    require(np.__version__ == plan["numpy"], "Unreviewed NumPy version")
    def git_blob(relative):
        return subprocess.check_output(["git", "-C", str(REPO), "show",
            plan["gpu_commit"] + ":research/wakeword_p4/" + relative])
    helper_bytes = git_blob("training/us_coverage.py")
    precheck_bytes = git_blob("results/4090-v45-sampler-precheck-v1.json")
    require(sha(helper_bytes) == plan["helper_sha256"], "Original helper differs")
    require(sha(precheck_bytes) == plan["gpu_precheck_sha256"], "GPU precheck differs")
    precheck = json.loads(precheck_bytes)
    pack = args.pack.resolve()
    report_bytes = (pack / "report.json").read_bytes()
    training_bytes = (pack / "training.npz").read_bytes()
    require(sha(report_bytes) == plan["pack_report_sha256"], "Original feature report differs")
    require(sha(training_bytes) == plan["training_npz_sha256"], "Original training NPZ differs")
    report = json.loads(report_bytes)
    rows = report["records"]
    require(len(rows) == report["included"] == report["attempted_variants"] == plan["variants"],
            "Incomplete original training population")
    with np.load(io.BytesIO(training_bytes), allow_pickle=False) as arrays:
        x, y = arrays["x"], arrays["y"]
    require(list(x.shape) == plan["shape"] and x.dtype == np.uint16
            and y.shape == (plan["variants"],) and y.dtype == np.float32, "NPZ geometry differs")
    raw_arrays, complete_ends, source_groups, paths = {}, {}, {}, set()
    for index, row in enumerate(rows):
        require(row["pack_index"] == index and row["split"] == "training"
                and row["mechanical_pass"] and row["entire_signal_fits"], "Invalid training identity")
        require(y[index] == int(row["target"] == "hey_nyx"), "Training label differs")
        relative = Path(row["raw"])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() not in paths, "Invalid or duplicate raw path")
        paths.add(relative.as_posix())
        raw_bytes = (pack / relative).read_bytes()
        require(sha(raw_bytes) == row["raw_file_sha256"], "Original raw file differs")
        raw = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        require(raw.dtype == np.uint16 and list(raw.shape) == row["raw_shape"]
                and raw.ndim == 2 and raw.shape[1] == 40
                and sha(raw.tobytes()) == row["raw_sha256"], "Original raw data differs")
        # Independent sample-ledger enumeration, not a classifier or phone-end claim.
        legal = range(150, len(raw) + 1)
        full = [end for end in legal if (end - 150) * 160 <= row["lead_samples"]
                and (end + 1) * 160 >= row["lead_samples"] + row["signal_samples"]]
        require(full and full == list(range(row["feature_crop_end"], min(350, len(raw)) + 1)),
                "Complete-container endpoint ledger differs")
        low = full[0]
        np.testing.assert_array_equal(raw[low - 150:low], x[index])
        raw_arrays[index], complete_ends[index] = raw, full
        source_groups.setdefault((row["target"], row["voice"]), []).append(index)
    phrases = ["hey_nyx", "hey_nick", "hey_next", "hey_mix"]
    presets = ["lessac", "ryan", "ljspeech"]
    require(set(source_groups) == {(phrase, voice) for phrase in phrases for voice in presets}
            and all(len(indices) == 27 for indices in source_groups.values()), "Training grid differs")
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_bytes(plan_bytes)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    helper = args.output / "original_us_coverage.py"
    helper.write_bytes(helper_bytes)
    spec = importlib.util.spec_from_file_location("original_v45_us_coverage", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual, fixed = module.USCoverage(pack, temporal=True), module.USCoverage(pack, temporal=False)
    source_rng = np.random.default_rng(plan["source_seed"])
    endpoint_rng = np.random.default_rng(plan["endpoint_seed"])
    slots = plan["positive_slots"] + plan["complete_negative_slots"] + plan["timeline_negative_slots"]
    untouched = [slot for slot in range(128) if slot not in slots]
    require(len(slots) == 20 and len(untouched) == 108, "Wrong slot plan")
    baseline = np.arange(128 * 150 * 40, dtype=np.float32).reshape(128, 150, 40)
    baseline *= np.float32(26 / (baseline.size - 1))
    baseline_sha = sha(baseline.tobytes())
    trace, counts = [], Counter()
    for batch in range(plan["batches"]):
        actual_batch, fixed_batch = actual.replace(baseline), fixed.replace(baseline)
        np.testing.assert_array_equal(actual_batch[untouched], baseline[untouched])
        np.testing.assert_array_equal(fixed_batch[untouched], baseline[untouched])
        require(sha(baseline.tobytes()) == baseline_sha, "Input sentinel tensor was mutated")
        for position, slot in enumerate(slots):
            phrase = "hey_nyx" if position < 8 else str(source_rng.choice(phrases[1:]))
            voice = str(source_rng.choice(presets))
            index = int(source_rng.choice(sorted(source_groups[phrase, voice])))
            counts[voice + "/" + phrase] += 1
            raw, full = raw_arrays[index], complete_ends[index]
            lower, upper = (full[0], full[-1]) if position < 14 else (150, len(raw))
            end = int(endpoint_rng.integers(lower, upper + 1))
            expected = np.clip(raw[end - 150:end].astype(np.float32) / 25.6, 0, 26)
            np.testing.assert_array_equal(actual_batch[slot], expected)
            np.testing.assert_array_equal(fixed_batch[slot], np.clip(x[index].astype(np.float32) / 25.6, 0, 26))
            trace.append(dict(batch=batch, slot=slot, index=index, end=end,
                mode="complete" if position < 14 else "timeline", tensor_sha256=sha(expected.tobytes())))
        require(actual.rng.bit_generator.state == fixed.rng.bit_generator.state == source_rng.bit_generator.state,
                "Original source RNG sequence changed")
        require(actual.temporal_rng.bit_generator.state == endpoint_rng.bit_generator.state,
                "Endpoint RNG sequence changed")
    require(len(trace) == plan["batches"] * plan["draws_per_batch"] == 1000
            and dict(counts) == precheck["counts"], "Original GPU source-choice counts differ")
    trace_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in trace).encode()
    (args.output / "draws.jsonl").write_bytes(trace_bytes)
    result = dict(scope=plan["scope"], all_checks_pass=True, variants=len(rows),
        raw_file_and_data_hashes_authenticated=len(raw_arrays), complete_endpoint_ledgers_exact=len(complete_ends),
        original_low_crops_exact=len(rows), batches=plan["batches"], source_and_endpoint_draws=len(trace),
        sampled_temporal_tensors_exact=len(trace), fixed_source_tensors_exact=len(trace),
        untouched_slots_per_batch=len(untouched), original_input_tensor_unchanged=True,
        original_source_rng_and_new_endpoint_rng_exact=True, original_gpu_source_counts_exact=True,
        counts=dict(counts), source_seed=plan["source_seed"], endpoint_seed=plan["endpoint_seed"],
        gpu_commit=plan["gpu_commit"], plan_sha256=sha(plan_bytes), script_sha256=sha(Path(__file__).read_bytes()),
        helper_sha256=sha(helper_bytes), gpu_precheck_sha256=sha(precheck_bytes),
        training_npz_sha256=sha(training_bytes), pack_report_sha256=sha(report_bytes),
        trace_sha256=sha(trace_bytes), host=dict(python=platform.python_version(), numpy=np.__version__,
        system=platform.system(), machine=platform.machine()), limits=plan["limits"])
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
