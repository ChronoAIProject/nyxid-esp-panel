"""Score the pinned training-only gain and reproduce every frozen C161 control."""
from pathlib import Path

import numpy as np
import soundfile as sf

from fit_c161_history_gain_v1 import FIT, MODEL, MODEL_SHA, RESULTS, STEM, apply_gain, frontend, read, save, score, sha
from review_c161_equal_length_history_v1 import PAIRED, SCORES, SOURCE
from run_parallel_context_streams_v1 import material


def replay():
    fit = read(FIT)
    assert sha(Path(__file__).with_name("fit_c161_history_gain_v1.py")) == fit["code_sha256"]
    assert sha(MODEL) == fit["model_sha256"] == MODEL_SHA
    weights = np.asarray(fit["weights"])
    original = read(PAIRED)
    assert sha(SCORES) == original["scores_sha256"]
    train_hashes = set()
    for row in fit["samples"]:
        train_hashes.add(row["sha256"])
        train_hashes.update(h["sha256"] for h in row["history"])
    source = read(SOURCE)
    traces, rows, streams = {}, [], []
    with np.load(SCORES, allow_pickle=False) as reference:
        def checked(pcm, key):
            x, clock = frontend(pcm)
            control, old = score(x, clock)
            np.testing.assert_array_equal(clock, reference[key + "_clock"])
            np.testing.assert_array_equal(control, reference[key + "_c161"])
            new, events = score(apply_gain(x, weights), clock)
            traces[key] = new
            return dict(c161=old, gain=events)

        for stream in source["streams"]:
            name = stream["name"]
            pcm, timeline = material(stream, source["gap_samples"])
            stream_events = checked(pcm, name + "_stream")
            streams.append(dict(name=name, events=stream_events))
            items = {item["text_index"]:item for item in stream["items"]}
            for clip in timeline:
                if clip["kind"] != "clip":
                    continue
                index, start, end = clip["text_index"], clip["start"], clip["end"]
                item = items[index]
                assert source["files"][item["wav"]] not in train_hashes
                target, rate = sf.read(item["wav"], dtype="int16")
                assert rate == 16000 and np.array_equal(target, pcm[start:end])
                isolated = checked(target, f"{name}_{index}")
                continuous = {arm:[e for e in ev if start < e <= end] for arm, ev in stream_events.items()}
                rows.append(dict(stream=name, index=index, start=start, end=end, text=clip["text"], group=clip["group"],
                                 pcm_sha256=item["pcm_sha256"], isolated=isolated, continuous=continuous))
    assert len(rows) == 144 and len(traces) == 147
    archive = RESULTS / f"{STEM}-development-scores-v1.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **traces)
    save(RESULTS / f"{STEM}-development-v1.json",
         dict(fit_sha256=sha(FIT), streams_plan_sha256=sha(SOURCE), control_scores_sha256=sha(SCORES),
              gain_scores_sha256=sha(archive), streams=streams, records=rows))


if __name__ == "__main__":
    replay()
