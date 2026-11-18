"""C161 with fixed ambient-history restore, alone and with the original event path."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import webrtcvad

from c161_binary40_parallel_sources_v1 import inputs
from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, REF_KERNEL, RESULTS, THRESHOLD, read, save, score, sha
from replay_c161_ambient_history_v1 import BUILD, CPP, STEM
from replay_c161_binary40_parallel_events_v2 import merge
from replay_c161_history_gain_cohorts_v1 import COHORTS, REFERENCE
from uint8_event_policy_v1 import event_steps


def event_rows(steps, clock):
    return [dict(step=s, relative_sample=int(clock[s-1])) for s in steps]


def run():
    source = read(REFERENCE)
    build = read(BUILD / "build.json")
    assert sha(MODEL) == MODEL_SHA and sha(CPP) == build["cpp_sha256"]
    assert sha(BUILD / "frontend.so") == build["library_sha256"]
    binding = ctypes.CDLL(str((BUILD / "frontend.so").resolve()))
    native = binding.ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    native.restype = ctypes.c_int
    it = REF_KERNEL(model_path=str(MODEL))
    it.allocate_tensors()
    scale, zero = it.get_input_details()[0]["quantization"]
    for cohort in COHORTS:
        control = source["cohorts"][cohort]
        assert sha(Path(control["path"])) == control["sha256"]
        rows, traces = [], {}
        for case, old in zip(inputs(cohort), read(Path(control["path"]))["records"], strict=True):
            index, record, pcm, _, _, _, _, _ = case
            assert index == old["index"] and (record == old["source"] or cohort == "continuous")
            pad = -len(pcm) % 160
            padded = np.concatenate((np.zeros(PRIME, np.int16), pcm, np.zeros(pad, np.int16)))
            hops = padded.reshape(-1, 160)
            vad = webrtcvad.Vad(0)
            speech = np.fromiter((vad.is_speech(frame.tobytes(), 16000) for frame in hops), dtype=np.uint8, count=len(hops))
            raw = np.empty((len(hops)-2, 40), np.uint16)
            length = PRIME + len(pcm)
            key = f"{old['phase']}_{index}"
            for treatment in (0, 1):
                code = native(padded.ctypes.data, speech.ctypes.data, len(hops), treatment, raw.ctypes.data, raw.size)
                assert code == 0
                x, support, _ = prepare(raw, length)
                clock = np.minimum((support + 479) // 480 * 480, length) - PRIME
                if treatment == 0:
                    digest = hashlib.sha256()
                    for first in range(0, len(x), 1200):
                        q = np.clip(np.rint(x[first:first+1200] / scale) + zero, -128, 127).astype(np.int8)
                        digest.update(q.tobytes())
                    assert digest.hexdigest() == old["timing"]["quantized_input_sha256"]["c161"]
                    trace = Path(old["trace"]["path"])
                    assert sha(trace) == old["trace"]["sha256"]
                    with np.load(trace, allow_pickle=False) as z:
                        np.testing.assert_array_equal(clock, z["relative_block_return"])
                else:
                    codes, _ = score(x, clock)
                    adjusted = event_rows(event_steps(codes, THRESHOLD), clock)
                    traces[key] = codes
            emitted, suppressed = merge(dict(c161=old["events"]["c161"], ambient=adjusted))
            rows.append(dict(index=index, phase=old["phase"], c161=old["events"]["c161"], ambient=adjusted,
                             pair=emitted, suppressed=suppressed, trace_sha256=old["trace"]["sha256"]))
        assert len(rows) == len(traces) == control["rows"]
        archive = RESULTS / f"{STEM}-{cohort}-scores-v1.npz"
        with archive.open("xb") as handle:
            np.savez_compressed(handle, **traces)
        save(RESULTS / f"{STEM}-{cohort}-v1.json", dict(cohort=cohort, source_sha256=control["sha256"],
             model_sha256=MODEL_SHA, scores_sha256=sha(archive), build_sha256=sha(BUILD/"build.json"), records=rows))
        print(cohort, len(rows), flush=True)


if __name__ == "__main__":
    run()
