"""Replay the deployed C161 operating point and fixed input gain across eight consumed cohorts."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from pymicro_features import MicroFrontend
from scipy.signal import lfilter

from c161_binary40_parallel_sources_v1 import inputs
from fit_c161_history_gain_v1 import FIT, MODEL, PRIME, REF_KERNEL, RESULTS, STEM, THRESHOLD, apply_gain, features, read, save, sha
from uint8_event_policy_v1 import event_steps

REFERENCE = RESULTS / "4090-c161-binary40-parallel-runtime-original-v1.json"
COHORTS = ("builtin", "daily", "joe", "mac", "continuous", "primed", "jenny", "melo")


def blocks(pcm):
    frontend = MicroFrontend()
    pending = np.empty((0, 40), dtype=np.uint16)
    # PCM splits and prime are multiples of the 160-sample hop, except at true EOF.
    for source in (np.zeros(PRIME, np.int16), pcm):
        for offset in range(0, len(source), 48000):
            raw = np.concatenate((pending, features(source[offset:offset+48000], frontend)))
            count = len(raw) // 3 * 3
            pending = raw[count:]
            if count:
                yield raw[:count]
    if len(pending):
        yield np.pad(pending, ((0, 3 - len(pending)), (0, 0)))


def detect(pcm, weights):
    model = REF_KERNEL(model_path=str(MODEL))
    model.allocate_tensors()
    inp, = model.get_input_details()
    out, = model.get_output_details()
    scale, zero = inp["quantization"]
    assert out["quantization"] == (1/256, 0)
    digest = hashlib.sha256()
    codes = []
    prev = np.zeros((1, 40))
    state = np.zeros((1, 40))
    for block in blocks(pcm):
        x = np.clip(block.astype(np.float32) / 25.6, 0, 26)
        baseline = np.clip(np.rint(x / scale) + zero, -128, 127).astype(np.int8)
        digest.update(baseline.tobytes())
        normalized = x.astype(np.float64) / 26
        averaged, state = lfilter([0.01], [1.0, -0.99], normalized, axis=0, zi=state)
        past = np.concatenate((prev, averaged[:-1]))
        prev = averaged[-1:]
        delta = normalized * weights[:, 0] + (normalized * past) * weights[:, 1]
        corrected = np.clip(x + 26 * delta, 0, 26).astype(np.float32)
        if not codes:
            np.testing.assert_array_equal(corrected, apply_gain(x, weights))
        q = np.clip(np.rint(corrected / scale) + zero, -128, 127).astype(np.int8)
        for start in range(0, len(q), 3):
            model.set_tensor(inp["index"], q[None, start:start+3])
            model.invoke()
            codes.append(model.get_tensor(out["index"])[0, 0])
    return digest.hexdigest(), np.asarray(codes, np.uint8)


def cohort(name):
    weights = np.asarray(read(FIT)["weights"])
    assert sha(MODEL) == read(FIT)["model_sha256"]
    spec = read(REFERENCE)["cohorts"][name]
    assert sha(Path(spec["path"])) == spec["sha256"]
    rows, arrays = [], {}
    for case, ref in zip(inputs(name), read(Path(spec["path"]))["records"], strict=True):
        index, source, pcm, _, _, _, _, timeline = case
        assert index == ref["index"]
        assert source == ref["source"] or (name == "continuous" and source["gap_samples"] == index)
        trace = Path(ref["trace"]["path"])
        assert sha(trace) == ref["trace"]["sha256"]
        digest, codes = detect(pcm, weights)
        assert digest == ref["timing"]["quantized_input_sha256"]["c161"]
        with np.load(trace, allow_pickle=False) as z:
            clock = z["relative_block_return"]
            assert len(codes) == len(clock)
        events = [int(clock[s-1]) for s in event_steps(codes, THRESHOLD)]
        key = f"{ref['phase']}_{index}"
        assert key not in arrays
        arrays[key] = codes
        rows.append(dict(index=index, phase=ref["phase"], source=ref["source"], samples=len(pcm),
                         trace_sha256=ref["trace"]["sha256"], c161=[e["relative_sample"] for e in ref["events"]["c161"]],
                         gain=events, timeline=timeline))
    assert len(rows) == spec["rows"]
    archive = RESULTS / f"{STEM}-{name}-scores-v1.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    save(RESULTS / f"{STEM}-{name}-v1.json", dict(cohort=name, fit_sha256=sha(FIT), records=rows,
                                                  reference_sha256=spec["sha256"], scores_sha256=sha(archive)))
    print(name, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", nargs="*", choices=COHORTS)
    for item in parser.parse_args().cohort:
        cohort(item)
