"""Rescore every eligible word proposal on the original 28.001 hours with both frozen INT8 heads."""
import ctypes
from pathlib import Path

import numpy as np
import soundfile as sf

import check_c161_polling_background as previous
import score_c161_paired_rooms as paired_score
from fit_c161_history_gain_v1 import PRIME, RESULTS, read, save, sha

poll = previous.poll


def main():
    models, hashes = paired_score.verifiers()
    path = RESULTS / (previous.check.STEM + "-background.json")
    report = read(path)
    word_path = RESULTS / (previous.CANDIDATE + "-v1.tflite")
    assert hashes["prior"] == report["model_sha256"] and sha(word_path) == report["word_sha256"]
    lib = previous.BUILD / "frontend.so"
    assert sha(lib) == report["native_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    rows = {}
    for cohort in ("melo", "primed", "jenny"):
        for source in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if source["phase"] != "background":
                continue
            key, src = f"{cohort}_{source['index']}", source["source"]
            old = report["records"][key]
            assert source["c161"] == [] and src["pcm_sha256"] == old["pcm_sha256"] and src["samples"] == old["samples"]
            ends = [p+previous.check.WAIT for p in old["proposals"] if 0 < p and p+previous.check.WAIT <= old["samples"]]
            assert ends == [c["available"] for c in old["checks"]]
            checks = []
            if ends:
                with sf.SoundFile(src.get("wav_path") or src["path"]) as f:
                    assert f.samplerate == 16000 and f.channels == 1 and len(f) == src["samples"]
                    full = np.zeros(ends[-1]+PRIME, np.int16)
                    f.read(out=full[PRIME:])
                hops = len(full)//160
                raw = np.empty((hops-2, 40), np.uint16)
                quiet = np.zeros(hops, np.uint8)
                assert native(full.ctypes.data, quiet.ctypes.data, hops, 0, raw.ctypes.data, raw.size) == 0
                for end, prior in zip(ends, old["checks"], strict=True):
                    patch = np.rint(poll.fit.window(raw, end).clip(0,666)*(255./666)).astype(np.uint8)
                    assert poll.digest(models["prior"].table[patch]) == prior["input_sha256"]
                    codes = {k:vm.invoke(patch).tolist() for k,vm in models.items()}
                    assert codes["prior"] == prior["codes"]
                    checks.append(dict(proposal=prior["proposal"], end=end, patch_sha256=poll.digest(patch), codes=codes))
            routes = {k:previous.check.merged([], [c["end"] for c in checks if c["codes"][k][0] >= 0]) for k in models}
            assert routes["prior"] == old["confirmed"]
            rows[key] = dict(pcm_sha256=src["pcm_sha256"], samples=src["samples"], proposals=old["proposals"],
                             checks=checks, events=routes)
    assert set(rows) == set(report["records"]) and sum(r["samples"] for r in rows.values()) == report["total_samples"]
    save(RESULTS / (paired_score.paired.STEM + "-background.json"), dict(records=rows, total_samples=report["total_samples"],
         prior_sha256=sha(path), native_sha256=sha(lib), word_sha256=sha(word_path), model_sha256=hashes,
         script_sha256=sha(Path(__file__)), policy="Same 66 consumed recordings, C161 clocks and all word proposals. Verify the original room-head input digest and all five output codes at every eligible window. Confirm 360 ms after proposal and apply the original merged cooldown."))


if __name__ == "__main__":
    main()
