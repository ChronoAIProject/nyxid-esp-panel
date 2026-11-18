"""Run all three actual INT8 binaries with both native C161 frontends."""
import ctypes
import hashlib
import json
import sys
from unittest.mock import patch

import numpy as np
import soundfile as sf

import verify_c161_ambient_stream_v1 as live
from audit_c161_history_paths_v1 import arbitrate, decode, times
from c161_binary40_parallel_sources_v1 import inputs
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, sha
from generate_c161_vowel_name_pair_v1 import AUDIO, SOURCE
from review_c161_equal_length_history_v1 import SOURCE as CONTEXT, SCORES
from run_parallel_context_streams_v1 import material

STEM = "4090-c161-vowel-name"


def cases(cohort):
    if cohort == "generated":
        src = read(SOURCE)
        assert sha(AUDIO) == src["pcm_sha256"]
        with np.load(AUDIO) as pcm:
            for r in src["records"]:
                wave = pcm[r["key"]]
                assert hashlib.sha256(wave.tobytes()).hexdigest() == r["pcm_sha256"]
                yield {k:r[k] for k in ("key","role","voice","kind","text","speed")}, np.pad(wave, (0,8000))
    elif cohort == "context":
        p = read(CONTEXT)
        for stream in p["streams"]:
            pcm, timeline = material(stream, p["gap_samples"])
            name = stream["name"]
            yield dict(key=name+"_stream", stream=name, timeline=timeline), pcm
            for span in timeline:
                if span["kind"] == "clip":
                    yield dict(key=f"{name}_{span['text_index']}", stream=name, **span), pcm[span["start"]:span["end"]]
    elif cohort == "willow":
        for i,r in enumerate(read(RESULTS/"4090-c161-willow-matched-scenes-run-plan-v1.json")["records"]):
            pcm, rate = sf.read(r["path"], dtype="int16")
            assert rate == 16000 and hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
            yield dict(key=str(i), voice=r["voice"], role=r["target_id"], text=r["carrier"],
                       start=r["target_start"], end=r["samples"]), pcm
    else:
        for i, source, pcm, *_ in inputs(cohort):
            meta = {k:source[k] for k in ("text","target","group","voice","id") if k in source}
            yield dict(key=str(i), **meta), pcm


def binding():
    lib = live.BUILD / "stream.so"
    assert sha(lib) == read(RESULTS/"4090-c161-ambient-history-pair-live-v1.json")["build"]["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve()))
    ptr = ctypes.c_void_p
    native.make_stream.argtypes = []
    native.make_stream.restype = ptr
    native.stream_hop.argtypes = [ptr, ptr, ctypes.c_int, ptr]
    native.stream_events.argtypes = [ptr, ctypes.c_uint8, ctypes.c_uint8]
    native.destroy_stream.argtypes = [ptr]
    native.destroy_stream.restype = None
    return native


def run(cohort):
    source = read(RESULTS / f"{STEM}-training-v1.json")
    assert sha(MODEL) == MODEL_SHA == source["baseline_sha256"]
    paths = {"c161": MODEL}
    for r in source["reports"]:
        p = RESULTS / r["path"]
        assert sha(p) == r["sha256"]
        paths[r["name"]] = p
    native = binding()
    rows, arrays = [], {}
    for meta, pcm in cases(cohort):
        row = dict(**meta, samples=len(pcm), pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), events={})
        key = meta["key"]
        for name, path in paths.items():
            with patch.object(live, "MODEL", path):
                codes, clock, emitted, suppressed = live.run_stream(native, pcm)
            c, a = np.asarray(codes, np.uint8)
            ck = np.asarray(clock)
            e1, e2 = decode(c, ck), decode(a, ck)
            assert arbitrate(e1, e2) == (emitted, suppressed)
            if name == "c161":
                arrays[key+"_clock"] = ck
            else:
                np.testing.assert_array_equal(ck, arrays[key+"_clock"])
            arrays[key+"_"+name] = np.stack([c,a])
            row["events"][name] = dict(original=times(e1), ambient=times(e2), pair=times(emitted))
        rows.append(row)
    if cohort == "context":
        with np.load(SCORES) as orig, np.load(RESULTS/"4090-c161-ambient-history-scores-v1.npz") as alt:
            for row in rows:
                k = row["key"]
                np.testing.assert_array_equal(arrays[k+"_c161"][0], orig[k+"_c161"])
                np.testing.assert_array_equal(arrays[k+"_c161"][1], alt[k+"_scores"])
                np.testing.assert_array_equal(arrays[k+"_clock"], orig[k+"_clock"])
    prefix = RESULTS/f"{STEM}-{cohort}"
    out = prefix.with_name(prefix.name+"-scores-v1.npz")
    with out.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with prefix.with_name(prefix.name+"-v1.json").open("x") as f:
        json.dump(dict(cohort=cohort, models={k:sha(p) for k,p in paths.items()}, scores_sha256=sha(out), records=rows), f, indent=2)
        f.write("\n")
    print(cohort, len(rows), flush=True)


if __name__ == "__main__":
    for selected in sys.argv[1:]:
        run(selected)
