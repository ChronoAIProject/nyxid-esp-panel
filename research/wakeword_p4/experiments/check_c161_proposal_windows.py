"""Compare two actual INT8 verifiers at the same causal proposals, including the complete background ledger."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
import soundfile as sf

from check_c161_event_transcripts_v1 import choose, source_pcm
from check_c161_midlevel_pocketsphinx import FRESH, ROOMS, STEM as LEXICAL
from check_c161_room_word_background import infer, native_features
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_proposal_windows import PARENT, STEM


def speech_pcm(rows):
    wanted = {(r["cohort"], r["key"]) for r in rows}
    for cohort in ("forward", "context", "daily", "joe", "mac", "bot", "transfer"):
        for key, pcm in source_pcm(cohort):
            if (cohort, key) in wanted:
                yield (cohort, key), pcm
    for cohort, path in (("new", FRESH), ("room", ROOMS)):
        for i, row in enumerate(read(path)["records"]):
            yield (cohort, str(i)), load_wav(row)


def background_pcm(rows):
    sources = {}
    for cohort in ("melo", "primed", "jenny"):
        for r in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if r["phase"] == "background":
                sources[f"{cohort}_{r['index']}"] = r["source"]
    for r in rows:
        src = sources.pop(r["key"])
        assert r["pcm_sha256"] == src["pcm_sha256"] and r["samples"] == src["samples"]
        pcm = None
        if r["checked"]:
            pcm, rate = sf.read(src.get("wav_path") or src["path"], dtype="int16")
            assert rate == 16000 and pcm.shape == (r["samples"],)
        yield (r["cohort"], r["key"]), pcm
    assert not sources


def main(background):
    models, identities = {}, {}
    for name, stem in (("parent", PARENT), ("trained", STEM)):
        path = RESULTS / f"{stem}-v1.tflite"
        identities[name] = sha(path)
        assert identities[name] == read(RESULTS / f"{stem}-export-v1.json")["sha256"]
        models[name] = REF_KERNEL(model_path=str(path))
        models[name].allocate_tensors()
    lib = BUILD / "frontend.so"
    assert sha(lib) == read(BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    suffix = "-background" if background else ""
    reference = RESULTS / f"{LEXICAL}{suffix}-v1.json"
    rows = read(reference)["records"]
    by_key = {(r["cohort"], r["key"]): r for r in rows}
    result = []
    parent_path = RESULTS / (f"{PARENT}-background-scores.npz" if background else f"{PARENT}-replay-scores-v1.npz")
    prior = RESULTS / (f"{PARENT}-background.json" if background else f"{PARENT}-replay-v1.json")
    assert sha(parent_path) == read(prior)["scores_sha256"]
    with np.load(parent_path) as saved:
        for key, pcm in (background_pcm(rows) if background else speech_pcm(rows)):
            r = by_key.pop(key)
            if pcm is not None:
                assert digest(pcm) == r["pcm_sha256"]
            times = sorted(set(r["proposals"]["midlevel"] + r["proposals"]["expanded"]))
            assert times == [x["proposal"] for x in r["checked"]]
            scores = {name: [] for name in models}
            if times:
                # Keep physical block delivery and the true EOF for partial final blocks.
                stop = min(len(pcm), ((times[-1] + 479) // 480 + 2) * 480)
                values, clock = native_features(native, pcm[:stop])
                indices = np.searchsorted(clock, times)
                np.testing.assert_array_equal(clock[indices], times)
                scores = {name: infer(model, values, indices).tolist() for name, model in models.items()}
                witness = r["key"] + "_mined" if background else "_".join(key) + "_codes"
                if witness in saved:
                    np.testing.assert_array_equal(scores["parent"], saved[witness][indices])
            events = {name: choose(r["baseline"], [t for t, value in zip(times, codes, strict=True) if value >= 235])
                      for name, codes in scores.items()}
            result.append(dict(cohort=key[0], key=key[1], pcm_sha256=r["pcm_sha256"], proposals=times,
                               scores=scores, events=events))
    assert not by_key
    save(RESULTS / f"{STEM}{suffix}-v1.json", dict(models=identities, source_sha256=sha(reference),
         cutoff=235, previous_scores_sha256=sha(parent_path), records=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background", action="store_true")
    main(parser.parse_args().background)
