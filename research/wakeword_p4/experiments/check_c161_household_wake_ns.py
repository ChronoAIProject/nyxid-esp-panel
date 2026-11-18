"""Test the existing P4 noise suppressor before frozen ordinary-speech detectors."""
from pathlib import Path

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from c161_v5_household_rooms import STEM as HOUSE
from check_c161_room_word import detectors, events
from check_c161_v5_command_ns import BUILD, digest, frames, replay
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from replay_c161_word_window_v1 import score
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_mined_room_word import PREFIX
from verify_c161_ambient_stream_v1 import run_stream

STEM = "4090-c161-household-wake-ns"
OUT = Path(".local/wakeword/c161-first-command-v5/ordinary-ns")


def main():
    source = RESULTS / f"{HOUSE}-v1.json"
    previous = RESULTS / f"{PREFIX}-hard-comparison-v1.json"
    ns = RESULTS / "4090-c161-v5-command-ns.json"
    records = read(source)["records"]
    rows = [(dict(r, ns_output_start=0), load_wav(r)) for r in records if r["role"] != "v5"]
    dry = {r["pcm_sha256"]: r for r in read(previous)["household"]}
    assert len(rows) == 72 and sha(BUILD / "nyx_first_command.elf") == read(ns)["elf_sha256"]
    assert sha(MODEL) == MODEL_SHA
    replay(rows, OUT)
    native = binding()
    models = list(detectors(f"{PREFIX}-hard"))
    scored, traces = [], {}
    for group in range(0, len(rows), 3):
        outputs = frames(OUT / f"{group}.log", rows, group)
        for index, (origin, filtered, allocated) in enumerate(outputs, start=group):
            case, original = rows[index]
            assert origin == 0 and digest(original) == case["pcm_sha256"]
            # Keep the NS device clock and its measured 96-sample delay. No look-ahead alignment.
            filtered = filtered[:len(original)]
            identity = digest(filtered)
            codes, clock, _, _ = run_stream(native, filtered)
            c161 = np.array(codes[0], np.uint8)
            signatures = [score(it, filtered, inp, out) for it, inp, out in models]
            for _, t in signatures:
                np.testing.assert_array_equal(t, clock)
            clean, mined = (pair[0] for pair in signatures)
            clock = np.array(clock)
            base = times(decode(c161, clock))
            merged = events(base, clean, mined, clock)
            key = str(index)
            traces.update({key + "_c161": c161, key + "_clean": clean, key + "_mined": mined, key + "_clock": clock})
            scored.append(dict(index=index, role=case["role"], voice=case["voice"], scene=case["scene"],
                               snr=case["command_snr_db"], start=case["start"], command_at=case["command_at"],
                               input_sha256=digest(original), filtered_sha256=identity, allocated_bytes=allocated,
                               dry=dry[case["pcm_sha256"]]["events"], ns=merged))
        print("scored", group + len(outputs), flush=True)
    archive = RESULTS / f"{STEM}-scores.npz"
    with archive.open("xb") as f:
        np.savez_compressed(f, **traces)
    save(RESULTS / f"{STEM}.json", dict(source_sha256=sha(source), comparison_sha256=sha(previous),
         ns_report_sha256=sha(ns), model_sha256=MODEL_SHA, scores_sha256=sha(archive),
         clocks="original 16 kHz emitted sample indices, with no NS delay subtracted", records=scored))


if __name__ == "__main__":
    main()
