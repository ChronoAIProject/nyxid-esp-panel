"""Replay the signed single-output model on all 66 previously used background recordings."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import webrtcvad

from audit_c161_history_paths_v1 import arbitrate, decode, times
from collect_multispeaker_formal_v1 import predict, prepare
from fit_c161_history_gain_v1 import PRIME, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from replay_c161_history_gain_cohorts_v1 import REFERENCE

STEM = "4090-c161-signed-confuser"
CACHE = Path(".local/wakeword/c161-signed-background-v1")


def replay(native, model, source, ref, clock):
    path = Path(source.get("wav_path") or source["path"])
    pcm, rate = sf.read(path, dtype="int16")
    assert rate == 16000 and pcm.shape == (source["samples"],)
    assert hashlib.sha256(pcm.tobytes()).hexdigest() == source["pcm_sha256"]
    full = np.r_[np.zeros(PRIME, np.int16), pcm]
    frames = np.pad(full, (0, -len(full) % 160)).reshape(-1, 160)
    vad = webrtcvad.Vad(0)
    speech = np.fromiter((vad.is_speech(h.tobytes(), 16000) for h in frames), np.uint8, count=len(frames))
    raw = np.empty((len(frames) - 2, 40), np.uint16)
    results = []
    for arm in (0, 1):
        assert native(frames.ctypes.data, speech.ctypes.data, len(frames), arm, raw.ctypes.data, raw.size) == 0
        values, support, _ = prepare(raw, len(full))
        np.testing.assert_array_equal(np.minimum((support + 479) // 480 * 480, len(full)) - PRIME, clock)
        _, scores, meta = predict(values, str(model), REF_KERNEL)
        if arm == 0:
            assert meta["quantized_input_sha256"] == ref["timing"]["quantized_input_sha256"]["c161"]
        results.append(scores)
    return np.stack(results)


def main():
    CACHE.mkdir(exist_ok=True)
    fit = read(RESULTS / f"{STEM}-training-v1.json")["reports"][0]
    model = RESULTS / fit["path"]
    assert sha(model) == fit["sha256"]
    lib = BUILD / "frontend.so"
    assert sha(lib) == read(BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    checked = read(RESULTS / f"{STEM}-transfer-v1.json")
    traced = {r["pcm_sha256"]: r for r in checked["records"] if r["group"] == "broadcast"}
    archive = RESULTS / f"{STEM}-transfer-scores-v1.npz"
    assert checked["model_sha256"] == fit["sha256"] and sha(archive) == checked["scores_sha256"]
    rows, arrays, reports = [], {}, {}
    for cohort in ("primed", "jenny", "melo"):
        spec = read(REFERENCE)["cohorts"][cohort]
        report = Path(spec["path"])
        assert sha(report) == spec["sha256"]
        reports[cohort] = spec["sha256"]
        for ref in read(report)["records"]:
            if ref["phase"] != "background":
                continue
            source = ref["source"]
            key = f"{cohort}_{ref['index']}"
            trace = Path(ref["trace"]["path"])
            assert sha(trace) == ref["trace"]["sha256"]
            with np.load(trace) as z:
                clock = z["relative_block_return"]
            pcm_sha = source["pcm_sha256"]
            if pcm_sha in traced:
                previous = traced[pcm_sha]
                with np.load(archive) as z:
                    scores = z["broadcast_" + previous["key"]]
            else:
                checkpoint = CACHE / f"{key}-{fit['sha256'][:16]}-{pcm_sha[:16]}-{sha(Path(__file__))[:16]}.npz"
                if not checkpoint.exists():
                    scores = replay(native, model, source, ref, clock)
                    tmp = checkpoint.with_suffix(".partial")
                    with tmp.open("wb") as f:
                        np.savez_compressed(f, scores=scores)
                    tmp.replace(checkpoint)
                with np.load(checkpoint) as z:
                    scores = z["scores"]
            primary, auxiliary = [decode(c, clock) for c in scores]
            pair = times(arbitrate(primary, auxiliary)[0])
            if pcm_sha in traced:
                assert pair == traced[pcm_sha]["pair"] and times(primary) == traced[pcm_sha]["primary"]
            arrays[key] = scores
            rows.append(dict(key=key, source_id=source.get("source_id", Path(source.get("path", "")).stem),
                             samples=source["samples"], pcm_sha256=pcm_sha, clock_trace_sha256=ref["trace"]["sha256"],
                             primary=times(primary), auxiliary=times(auxiliary), pair=pair))
            print(key, len(rows), len(pair), flush=True)
    assert len(rows) == 66 and len({r["pcm_sha256"] for r in rows}) == len(rows)
    output = RESULTS / f"{STEM}-background-scores-v1.npz"
    with output.open("xb") as f:
        np.savez_compressed(f, **arrays)
    save(RESULTS / f"{STEM}-background-v1.json", dict(model_sha256=sha(model), source_reports=reports,
         native_sha256=sha(lib), transfer_sha256=sha(archive), scores_sha256=sha(output), records=rows))


if __name__ == "__main__":
    main()
