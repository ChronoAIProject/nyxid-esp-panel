"""Cross speech/silence pre-onset model input and target frontend features for all 144 clips.

Use the first unrounded three-frame support ending strictly after onset as
the split. This group may straddle onset. Crossed inputs are diagnostic only.
"""
import argparse
from functools import partial
import json
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter, OpResolverType
import numpy as np

from collect_multispeaker_formal_v1 import predict, prepare
from review_c161_equal_length_history_v1 import (
    MODEL, MODEL_SHA, PRIME, RESULTS, SCORES, SOURCE, THRESHOLD, features, sha,
)
from run_parallel_context_streams_v1 import material
from uint8_event_policy_v1 import event_steps

HISTORY = RESULTS / "4090-c161-equal-length-history-v1"


def frontend(pcm):
    full = np.concatenate((np.zeros(PRIME, np.int16), pcm))
    values, support, _ = prepare(features(full), len(full))
    return values, support


def run(output):
    source = json.loads(SOURCE.read_bytes())
    prior = json.loads((HISTORY / "review.json").read_bytes())
    assert sha(MODEL) == MODEL_SHA == prior["model_sha256"]
    assert sha(SOURCE) == prior["source_plan_sha256"] and sha(SCORES) == prior["physical_scores_sha256"]
    assert sha(HISTORY / "scores.npz") == prior["silence_scores_sha256"]
    rows = {(r["stream"], r["index"]): r for r in prior["records"]}
    runtime = partial(Interpreter, num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    traces, report = {}, []
    output.mkdir(parents=True, exist_ok=False)
    with np.load(HISTORY / "scores.npz", allow_pickle=False) as quiet_ref, np.load(SCORES, allow_pickle=False) as real_ref:
        for stream in source["streams"]:
            name = stream["name"]
            pcm, timeline = material(stream, source["gap_samples"])
            speech, _ = frontend(pcm)
            _, continuous_codes, _ = predict(speech, str(MODEL), runtime)
            np.testing.assert_array_equal(continuous_codes, real_ref[f"{name}_stream_c161"])
            for clip in timeline:
                if clip["kind"] != "clip":
                    continue
                index, start, end = clip["text_index"], clip["start"], clip["end"]
                key = f"{name}_{index}"
                row = rows[name, index]
                assert (start, end) == (row["start"], row["end"])
                edited = pcm[:row["edited_pcm_samples"]].copy()
                edited[:start] = 0
                quiet, support = frontend(edited)
                _, quiet_codes, _ = predict(quiet, str(MODEL), runtime)
                np.testing.assert_array_equal(quiet_codes, quiet_ref[key])
                clock = np.minimum((support + 479) // 480 * 480, len(edited) + PRIME) - PRIME
                split = 3 * int(np.searchsorted(support, PRIME + start, side="right"))
                conditions = {
                    "speech_prefix_quiet_target": (speech[:split], quiet[split:]),
                    "quiet_prefix_speech_target": (quiet[:split], speech[split:len(quiet)]),
                }
                crossed_events = {}
                for label, inputs in conditions.items():
                    _, codes, _ = predict(np.concatenate(inputs), str(MODEL), runtime)
                    steps = event_steps(codes, THRESHOLD)
                    crossed_events[label] = [int(clock[s - 1]) for s in steps if start < clock[s - 1] <= end]
                    traces[f"{key}_{label}"] = codes
                report.append(dict(stream=name, index=index, split_frame=split, **crossed_events))
    assert len(report) == len(rows) == 144 and len(traces) == 288
    archive = output / "scores.npz"
    np.savez_compressed(archive, **traces)
    data = dict(history_report_sha256=sha(HISTORY / "review.json"), crossed_scores_sha256=sha(archive), records=report)
    (output / "review.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for crossed traces and report")
    run(parser.parse_args().output)
