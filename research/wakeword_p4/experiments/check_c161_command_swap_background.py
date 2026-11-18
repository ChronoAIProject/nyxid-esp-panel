"""Verify joint notifications with exact INT8 at every tick capable of affecting the decoder."""
import ctypes
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from check_c161_room_word_background import CANDIDATE, infer, native_features
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from train_c161_command_swap import STEM


def active_indices(clean):
    # Every joint code is <= the clean UINT8. Below 587.5 the decoder re-arms
    # irrespective of the second classifier. Include every other complete five-tick window.
    sums = np.convolve(clean, np.ones(5, np.int32), mode="valid")
    required = np.convolve((sums * 2 >= 1175).astype(np.int8), np.ones(5, np.int8))
    return np.flatnonzero(required)


def main():
    base_file = RESULTS / f"{CANDIDATE}-background.json"
    base = read(base_file)
    scores = RESULTS / f"{CANDIDATE}-background-scores.npz"
    assert sha(scores) == base["scores_sha256"]
    lib = BUILD/"frontend.so"
    assert sha(lib) == base["frontend_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    models, identities, refs = {}, {}, {}
    for mode in ("own", "swapped"):
        file = RESULTS/f"{STEM}-{mode}-v1.tflite"
        identities[mode] = sha(file)
        assert identities[mode] == read(RESULTS/f"{STEM}-{mode}-export-v1.json")["sha256"]
        models[mode] = REF_KERNEL(model_path=str(file))
        models[mode].allocate_tensors()
        refs[mode] = {r["meta"]["pcm_sha256"]:r["events"]["both"] for r in
                      read(RESULTS/f"{STEM}-{mode}-comparison-v1.json")["records"] if r["cohort"] == "transfer" and r["meta"]["group"] == "broadcast"}
    records, arrays = {}, {}
    with np.load(scores) as prior:
        for cohort in ("melo","primed","jenny"):
            for r in read(RESULTS/f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
                if r["phase"] != "background":
                    continue
                key = f"{cohort}_{r['index']}"
                old = base["records"][key]
                src = r["source"]
                pcm, rate = sf.read(Path(src.get("wav_path") or src["path"]), dtype="int16")
                assert rate == 16000 and digest(pcm) == old["pcm_sha256"] == src["pcm_sha256"] and not r["c161"]
                clean = prior[key+"_clean"]
                features, clock = native_features(native, pcm)
                assert digest(clock) == old["clock_sha256"] and len(clock) == len(clean) == old["ticks"]
                indices = active_indices(clean)
                arrays[key+"_indices"] = indices.astype(np.int32)
                hits = {}
                for mode in ("parent", *models):
                    candidate = prior[key+"_mined"][indices] if mode == "parent" else infer(models[mode], features, indices)
                    # Fill with the known upper bound at unqueried ticks.
                    combined = clean.copy()
                    combined[indices] = np.minimum(clean[indices], candidate)
                    hits[mode] = times(decode(combined, clock))
                    if mode == "parent":
                        assert hits[mode] == old["events"]["both"]
                    else:
                        arrays[key+"_"+mode] = candidate
                        if old["pcm_sha256"] in refs[mode]:
                            assert hits[mode] == refs[mode].pop(old["pcm_sha256"])
                records[key] = dict(pcm_sha256=old["pcm_sha256"], source=old["source"], samples=len(pcm),
                                    ticks=len(clock), inferred_per_model=len(indices), events=hits)
    assert len(records) == 66 and not any(refs.values())
    archive = RESULTS/f"{STEM}-background-scores.npz"
    with archive.open("xb") as out:
        np.savez_compressed(out, **arrays)
    save(RESULTS/f"{STEM}-background.json", dict(baseline_sha256=sha(base_file), clean_scores_sha256=sha(scores),
         model_sha256=identities, frontend_sha256=sha(lib), sparse_scores_sha256=sha(archive), records=records))


if __name__ == "__main__":
    main()
