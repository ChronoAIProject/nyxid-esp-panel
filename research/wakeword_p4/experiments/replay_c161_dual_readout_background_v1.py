"""Check the deployable dual readout across the three consumed continuous background sets."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

from audit_c161_history_paths_v1 import arbitrate, decode, times
from check_c161_speech_history_gate_v1 import history_gate
from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import PRIME, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from replay_c161_history_gain_cohorts_v1 import REFERENCE

STEM = "4090-c161-dual-readout-background"


def run_file(ref, original, expected, native, model):
    source = ref["source"]
    path = Path(source.get("wav_path") or source["path"])
    pcm, rate = sf.read(path, dtype="int16")
    assert rate == 16000 and len(pcm) == source["samples"] and pcm.ndim == 1
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == source["pcm_sha256"]
    full = np.r_[np.zeros(PRIME, np.int16), pcm]
    frames = np.pad(full,(0,-len(full)%160)).reshape(-1,160)
    vad = webrtcvad.Vad(0)
    speech = np.fromiter((vad.is_speech(h.tobytes(),16000) for h in frames),np.uint8,count=len(frames))
    raw = np.empty((len(frames)-2,40),np.uint16)
    assert native(frames.ctypes.data,speech.ctypes.data,len(frames),1,raw.ctypes.data,raw.size)==0
    values,support,_ = prepare(raw,len(full))
    clock = np.minimum((support+479)//480*480,len(full))-PRIME
    trace = Path(ref["trace"]["path"])
    assert sha(trace) == ref["trace"]["sha256"] == original["trace_sha256"]
    with np.load(trace) as z:
        np.testing.assert_array_equal(clock,z["relative_block_return"])
    it = REF_KERNEL(model_path=str(model))
    it.allocate_tensors()
    inp, = it.get_input_details()
    out = it.get_output_details()
    scale, zero = inp["quantization"]
    q = np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8).reshape(-1,3,40)
    original_code, trained = np.empty(len(q),np.uint8), np.empty(len(q),np.uint8)
    for i,frame in enumerate(q):
        it.set_tensor(inp["index"],frame[None])
        it.invoke()
        original_code[i] = it.get_tensor(out[0]["index"])[0,0]
        trained[i] = it.get_tensor(out[1]["index"])[0,0]
    np.testing.assert_array_equal(original_code, expected)
    active, ends = history_gate(pcm,clock)
    primary = original["c161"]
    assert not primary and not original["pair"]
    hits = {}
    for mode, scores in (("original",original_code),("always",trained),("conditional",np.where(active,trained,original_code))):
        hits[mode] = times(arbitrate(primary,decode(scores,clock))[0])
    return dict(index=ref["index"], source_id=source.get("source_id",path.stem), samples=len(pcm),
                pcm_sha256=source["pcm_sha256"], clocks=len(clock), qualifying_intervals=len(ends),
                active_steps=int(active.sum()), events=hits), trained


def main():
    source = read(REFERENCE)
    dual = RESULTS/"4090-c161-dual-readout-v1.tflite"
    assert sha(dual) == read(RESULTS/"4090-c161-dual-readout-v1.json")["model_sha256"]
    lib = BUILD/"frontend.so"
    assert sha(lib) == read(BUILD/"build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    for cohort in ("primed","jenny","melo"):
        path = Path(source["cohorts"][cohort]["path"])
        assert sha(path) == source["cohorts"][cohort]["sha256"]
        prior = read(RESULTS/f"4090-c161-ambient-history-{cohort}-v1.json")
        scores = RESULTS/f"4090-c161-ambient-history-{cohort}-scores-v1.npz"
        assert sha(scores) == prior["scores_sha256"]
        arrays, rows = {}, []
        with np.load(scores) as z:
            for ref, old in zip(read(path)["records"], prior["records"], strict=True):
                if ref["phase"] == "background":
                    key = f'background_{ref["index"]}'
                    row, arrays[key] = run_file(ref,old,z[key],native,dual)
                    rows.append(row)
                    print(cohort,row["index"],row["events"],flush=True)
        out = RESULTS/f"{STEM}-{cohort}-scores-v1.npz"
        with out.open("xb") as f:
            np.savez_compressed(f,**arrays)
        save(RESULTS/f"{STEM}-{cohort}-v1.json",dict(model_sha256=sha(dual), source_report_sha256=sha(path),
             scores_sha256=sha(out), records=rows))


if __name__ == "__main__":
    main()
