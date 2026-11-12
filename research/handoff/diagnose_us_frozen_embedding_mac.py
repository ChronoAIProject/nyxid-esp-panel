"""Run the frozen training-only US diagnostic without touching audio devices."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import soundfile as sf


HERE = Path(__file__).resolve().parent
EMBEDDING = HERE.parent / "wakeword_p4/embedding"
PLAN = HERE / "plans/us-frozen-embedding-diagnostic-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def event_indices(scores, threshold, refractory, rearm):
    result, armed, last = [], True, -1000
    for index, score in enumerate(scores):
        if score < threshold * rearm and index - last >= refractory:
            armed = True
        if armed and score >= threshold and index - last >= refractory:
            result.append(index)
            last, armed = index, False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Authenticated original pack's audio directory")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Fresh output directory required")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    versions = {name: importlib.metadata.version(name) for name in plan["versions"]}
    require(versions == plan["versions"], "Frozen runtime differs")
    source = args.source.resolve()
    manifest_bytes = (source / "manifest.jsonl").read_bytes()
    require(sha(manifest_bytes) == plan["source_manifest_sha256"], "Original source manifest differs")
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    require(len(rows) == plan["source_clips"], "Incomplete training cohort")
    counts = Counter((r["voice"], r["target"]) for r in rows)
    require(dict(counts) == {(v, p): plan["clips_per_voice_phrase"]
                            for v in plan["voices"] for p in plan["phrases"]}, "Training grid differs")
    models = args.models.resolve()
    frozen_bytes = (models / plan["original_frozen_model_plan"]).read_bytes()
    require(sha(frozen_bytes) == plan["original_frozen_model_plan_sha256"], "Frozen model plan differs")
    frozen = json.loads(frozen_bytes)
    require(frozen["selected"]["selected"]["threshold"] == plan["threshold"], "Threshold changed")
    inputs = {}
    for name, expected in frozen["runtime_inputs"].items():
        inputs[name] = sha((models / name).read_bytes())
        require(inputs[name] == expected, "Runtime artifact differs: " + name)
    for name, expected in frozen["helpers"].items():
        require(sha((EMBEDDING / name).read_bytes()) == expected, "Frozen helper differs: " + name)
    paths = []
    for member in frozen["selected"]["members"]:
        name = member["checkpoint"] + "/head.tflite"
        paths.append(models / name)
        inputs[name] = sha(paths[-1].read_bytes())
        require(inputs[name] == member["head_sha256"], "Frozen head differs")
    sys.path.insert(0, str(EMBEDDING))
    import prepare_head_features as extractor
    from evaluate_equal_mean import EqualMeanHead, equal_mean
    from evaluate_head import events
    detector = EqualMeanHead(paths)
    extractor.initialize(models / "embedding-stream-int8-quiet.tflite",
                         models / "native-fft-window.npy", models / "native-mel-filter.npy")
    args.output.mkdir(parents=True)
    (args.output / "plan.json").write_bytes(plan_bytes)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "original-model-plan.json").write_bytes(frozen_bytes)
    started = time.monotonic()
    results, summaries = [], defaultdict(lambda: Counter())
    with (args.output / "rows.jsonl").open("x") as log:
        for order, row in enumerate(rows):
            require(row["order"] == order and row["split"] == "training"
                    and row["role"] == "us-coverage-training-only"
                    and row["quality_pass"] and not row["trimmed"] and not row["normalized"],
                    "Source training identity differs")
            relative = Path(row["path"])
            require(relative.parts == (relative.name,), "Unexpected original source path")
            path = source / relative
            require(sha(path.read_bytes()) == row["sha256"], "Original WAV differs")
            pcm, rate = sf.read(path, dtype="int16")
            require(rate == plan["sample_rate"] and pcm.ndim == 1
                    and len(pcm) == row["samples"]
                    and sha(pcm.astype("<i2").tobytes()) == row["pcm_sha256"], "Original PCM differs")
            rendered = np.pad(pcm, (plan["lead_zero_samples"], plan["tail_zero_samples"]))
            embeddings = extractor.extract(rendered)
            a, times = detector.members[0].scores(embeddings, plan["smoothing_steps"])
            b, other_times = detector.members[1].scores(embeddings, plan["smoothing_steps"])
            np.testing.assert_array_equal(times, other_times)
            mean = equal_mean(a, b)
            indices = event_indices(mean, plan["threshold"], plan["refractory_steps"], plan["rearm_fraction"])
            require(len(indices) == events(mean, plan["threshold"]), "Frozen event counter differs")
            ends = np.rint(times * plan["sample_rate"]).astype(np.int64)
            offsets = [int(ends[i] - plan["lead_zero_samples"]) for i in indices
                       if ends[i] <= len(rendered)]
            positive_end = (plan["lead_zero_samples"] + len(pcm)) / plan["sample_rate"]
            low, high = plan["positive_reference_window_seconds"]
            window = (times >= positive_end + low) & (times <= positive_end + high)
            filename = str(order) + "-arrays.npz"
            np.savez_compressed(args.output / filename, embeddings=embeddings,
                                member0=a, member1=b, mean=mean, times=times)
            result = dict(order=order, source=row["path"], voice=row["voice"], target=row["target"],
                          source_sha256=row["sha256"], source_pcm_sha256=row["pcm_sha256"],
                          source_samples=len(pcm), rendered_samples=len(rendered),
                          rendered_pcm_sha256=sha(rendered.astype("<i2").tobytes()),
                          arrays=filename, arrays_sha256=sha((args.output / filename).read_bytes()),
                          event_offsets_samples=offsets, priming_events=sum(s <= 0 for s in offsets),
                          any_after_source_start=any(s > 0 for s in offsets),
                          reference_window_max=float(mean[window].max()) if window.any() else 0.,
                          reference_window_high=bool(window.any() and np.any(mean[window] >= plan["threshold"])),
                          max_score=float(mean.max()))
            for key in (row["voice"] + "/" + row["target"], "all/" + row["target"]):
                summaries[key].update(clips=1, any_after_source_start=int(result["any_after_source_start"]),
                                      reference_window_high=int(result["reference_window_high"]),
                                      priming_events=result["priming_events"])
            log.write(json.dumps(result) + "\n")
            log.flush()
            results.append(result)
            if (order + 1) % 12 == 0:
                print(json.dumps(dict(completed=order + 1, total=len(rows),
                                      seconds=round(time.monotonic() - started, 1))), flush=True)
    report = dict(scope=plan["scope"], plan_sha256=sha(plan_bytes),
                  script_sha256=sha(Path(__file__).read_bytes()),
                  original_model_plan_sha256=sha(frozen_bytes),
                  source_manifest_sha256=sha(manifest_bytes), runtime_inputs_sha256=inputs,
                  frozen_threshold=plan["threshold"], versions=versions, clips=len(results),
                  summaries={k: dict(v) for k, v in sorted(summaries.items())},
                  rows_sha256=sha((args.output / "rows.jsonl").read_bytes()),
                  seconds=time.monotonic() - started, accepted=False,
                  limitations=plan["limitations"], decision=plan["decision"])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
