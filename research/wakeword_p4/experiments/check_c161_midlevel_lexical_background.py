"""Replay both midlevel clocks and bounded lexical confirmation over 66 complete recordings."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from calibrate_c161_pocketsphinx import OUT
from check_c161_midlevel_pocketsphinx import STEM, confirmed_record
from check_c161_pocketsphinx import setup
from check_c161_stream_stretch_v1 import run_quantized
from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import PRIME, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from replay_c161_history_gain_cohorts_v1 import REFERENCE

CACHE = Path(".local/wakeword/c161-pocket-reference/midlevel-background")


def quantized_stream(native, pcm, scale, zero):
    samples = len(pcm) + PRIME
    hops = np.zeros(((samples + 159) // 160, 160), np.int16)
    hops.ravel()[PRIME:samples] = pcm
    raw = np.empty((len(hops) - 2, 40), np.uint16)
    activity = np.zeros(len(hops), np.uint8)
    assert native(hops.ctypes.data, activity.ctypes.data, len(hops), 0, raw.ctypes.data, raw.size) == 0
    values, support, _ = prepare(raw, samples)
    clock = np.minimum((support + 479) // 480 * 480, samples) - PRIME
    q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
    return q, clock


def main():
    source = RESULTS / f"{STEM}-v1.json"
    run = read(source)
    model = RESULTS / run["model"]["path"]
    assert sha(model) == run["model"]["sha256"]
    scale, zero = REF_KERNEL(model_path=str(model)).get_input_details()[0]["quantization"]
    refs = read(REFERENCE)["cohorts"]
    decoder, cmn, spec = setup(10.0 ** -read(OUT)["selected_exponent"])
    assert spec == run["implementation"]
    library = BUILD / "frontend.so"
    assert sha(library) == read(BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(library.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    witnesses = {r["pcm_sha256"]: r for r in run["records"] if r["key"].startswith("broadcast_")}
    CACHE.mkdir(parents=True, exist_ok=True)
    signature = dict(speech_sha256=sha(source), frontend_sha256=sha(library), script_sha256=sha(Path(__file__)))
    rows = []
    for cohort in ("melo", "primed", "jenny"):
        path = Path(refs[cohort]["path"])
        assert sha(path) == refs[cohort]["sha256"]
        for r in read(path)["records"]:
            if r["phase"] != "background":
                continue
            src = r["source"]
            wav = Path(src.get("wav_path") or src["path"])
            key = f"{cohort}_{r['index']}"
            target = CACHE / f"{key}.json"
            if target.exists():
                cached = read(target)
                assert cached["signature"] == signature
                row = cached["record"]
            else:
                pcm, rate = sf.read(wav, dtype="int16")
                assert rate == 16000 and pcm.shape == (src["samples"],)
                assert hashlib.sha256(memoryview(pcm)).hexdigest() == src["pcm_sha256"]
                q, clock = quantized_stream(native, pcm, scale, zero)
                assert hashlib.sha256(memoryview(q)).hexdigest() == r["timing"]["quantized_input_sha256"]["c161"]
                assert not r["events"]["c161"]
                proposals = {}
                for name, expand in (("midlevel", False), ("expanded", True)):
                    codes, _ = run_quantized(q, model, expand)
                    proposals[name] = times(decode(codes, clock))
                row = confirmed_record(decoder, cmn, pcm, cohort, key, [], proposals)
                row.update(samples=len(pcm), source=src.get("source_id", wav.stem))
                tmp = target.with_suffix(".tmp")
                tmp.write_text(json.dumps(dict(signature=signature, record=row)))
                tmp.replace(target)
            assert row["key"] == key and row["pcm_sha256"] == src["pcm_sha256"]
            if row["pcm_sha256"] in witnesses:
                witness = witnesses.pop(row["pcm_sha256"])
                for field in ("proposals", "checked", "events"):
                    assert row[field] == witness[field]
            rows.append(row)
            print(len(rows), key, len(row["checked"]), len(row["events"]), flush=True)
    assert len(rows) == len({r["pcm_sha256"] for r in rows}) == 66 and not witnesses
    save(RESULTS / f"{STEM}-background-v1.json", dict(**signature, reference_sha256=sha(REFERENCE), records=rows))


if __name__ == "__main__":
    main()
