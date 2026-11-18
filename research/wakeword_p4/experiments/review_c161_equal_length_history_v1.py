"""Compare C161 on all 144 consumed context clips with speech or equal-length silence before each.

Use the original stream origin, 81600-sample prime, native return clock, full
target and fixed decoder. Preserve the next 480 samples when available (the
original gap). Reproduce the physical controls before the intervention. Save
every control and intervention score and the (start, end] attribution.
"""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

from ai_edge_litert.interpreter import Interpreter, OpResolverType
import numpy as np

from collect_multispeaker_formal_v1 import predict, prepare
from run_parallel_context_streams_v1 import material
from uint8_event_policy_v1 import event_steps

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT / "results"
sys.path.insert(0, str((ROOT / "training").resolve()))
from features import features

MODEL = Path(".local/wakeword/causal-source-balanced-int8-v1/int8.tflite")
MODEL_SHA = "c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81"
SOURCE = RESULTS / "4090-parallel-context-streams-plan-v1.json"
PAIRED = RESULTS / "4090-c161-bn-natural-language-original-v1.json"
SCORES = RESULTS / "4090-c161-bn-natural-language-scores-v1.npz"
THRESHOLD = 235 / 256
PRIME = 81600


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(pcm):
    full = np.concatenate((np.zeros(PRIME, dtype=np.int16), pcm))
    values, support, _ = prepare(features(full), len(full))
    interpreter = partial(Interpreter, num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    _, codes, _ = predict(values, str(MODEL), interpreter)
    clock = np.minimum(((support + 479) // 480) * 480, len(full)) - PRIME
    steps = event_steps(codes, THRESHOLD)
    return codes, clock, [int(clock[step - 1]) for step in steps]


def run(output):
    source = json.loads(SOURCE.read_bytes())
    paired = json.loads(PAIRED.read_bytes())
    assert sha(MODEL) == MODEL_SHA and sha(SCORES) == paired["scores_sha256"]
    assert source["gap_samples"] == 8000 and source["prime_samples"] == PRIME
    previous = {(r["stream"], r["index"]): r for r in paired["records"]}
    traces, rows = {}, []
    output.mkdir(parents=True, exist_ok=False)
    with np.load(SCORES, allow_pickle=False) as reference:
        for stream in source["streams"]:
            name = stream["name"]
            pcm, timeline = material(stream, source["gap_samples"])
            pcm_hashes = {item["text_index"]: item["pcm_sha256"] for item in stream["items"]}
            physical, clock, events = replay(pcm)
            np.testing.assert_array_equal(physical, reference[f"{name}_stream_c161"])
            np.testing.assert_array_equal(clock, reference[f"{name}_stream_clock"])
            for clip in timeline:
                if clip["kind"] != "clip":
                    continue
                index, start, end = clip["text_index"], clip["start"], clip["end"]
                prior = previous[name, index]
                assert (start, end) == (prior["continuous_clip_start"], prior["continuous_clip_end"])
                assert pcm_hashes[index] == prior["pcm_sha256"]
                key = f"{name}_{index}"
                edited = pcm[:min(end + 480, len(pcm))].copy()
                edited[:start] = 0
                codes, edited_clock, edited_events = replay(edited)
                # Keep all returns through clip end, including any true EOF flush.
                within = (clock > start) & (clock <= end)
                edited_within = (edited_clock > start) & (edited_clock <= end)
                np.testing.assert_array_equal(edited_clock[edited_within], clock[within])
                if start == 0:
                    np.testing.assert_array_equal(codes[edited_within], physical[within])
                actual = [t for t in events if start < t <= end]
                assert actual == [e["relative_sample"] for e in prior["continuous"]["c161"]]
                assert not any(t <= start for t in edited_events)
                traces[key] = codes
                rows.append(dict(stream=name, index=index, group=clip["group"], text=clip["text"],
                                 pcm_sha256=prior["pcm_sha256"], start=start, end=end,
                                 edited_pcm_samples=len(edited), physical_events=actual,
                                 silence_events=[t for t in edited_events if start < t <= end],
                                 all_silence_condition_events=edited_events))
    assert len(rows) == len(traces) == len(previous) == 144
    archive = output / "scores.npz"
    np.savez_compressed(archive, **traces)
    report = dict(model_sha256=MODEL_SHA, threshold=THRESHOLD, smoothing_steps=5,
                  source_plan_sha256=sha(SOURCE), paired_report_sha256=sha(PAIRED),
                  physical_scores_sha256=sha(SCORES), silence_scores_sha256=sha(archive),
                  prime_samples=PRIME, rate_hz=16000, records=rows)
    (output / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for score traces and report")
    run(parser.parse_args().output)
