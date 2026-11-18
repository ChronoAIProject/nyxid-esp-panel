"""Locate ordinary-voice misses on matched dry, room-only, and noisy household inputs."""
from zipfile import ZipFile

import numpy as np

from c161_v5_household_rooms import PLAN, STEM as HOUSE
from check_c161_household_quiet import acoustic_controls
from check_c161_room_word import detectors, events
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_word_window_v1 import score
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_mined_room_word import PREFIX


def main():
    stem = f"{PREFIX}-hard"
    quiet = RESULTS / f"{HOUSE}-quiet.json"
    rooms = RESULTS / f"{HOUSE}-v1.json"
    comparison = RESULTS / f"{stem}-comparison-v1.json"
    manifest = read(PLAN)
    archive = RESULTS / f"{HOUSE}-dry.zip"
    assert sha(archive) == manifest["waveform_archive_sha256"]
    baseline = read(quiet)
    assert baseline["reference_sha256"] == sha(rooms) == read(comparison)["household_sha256"]
    refs = {(r["voice"], r["scene"]): r for r in baseline["records"]}
    noisy = {r["pcm_sha256"]: r for r in read(comparison)["household"]}
    models = list(detectors(stem))
    rows = []
    with ZipFile(archive) as clips:
        for case in read(rooms)["records"]:
            if case["role"] != "ordinary" or case["command_snr_db"] != 12:
                continue
            ref = refs.pop((case["voice"], case["scene"]))
            controls = [*acoustic_controls(case, clips, manifest), ("noisy", load_wav(case))]
            for mode, pcm in controls:
                identity = digest(pcm)
                base = ref["noisy_events"] if mode == "noisy" else ref[mode]["original_events"]
                assert identity == (ref["noisy_pcm_sha256"] if mode == "noisy" else ref[mode]["pcm_sha256"])
                (dry, clock), (mined, other_clock) = [score(it, pcm, inp, out) for it, inp, out in models]
                np.testing.assert_array_equal(clock, other_clock)
                outputs = events(base, dry, mined, clock)
                if mode == "noisy":
                    assert outputs == noisy[identity]["events"]
                sums_window = (clock[4:] >= case["start"]) & (clock[4:] < case["command_at"])
                peaks = {k: int(np.convolve(codes, np.ones(5, dtype=int), "valid")[sums_window].max())
                         for k, codes in (("clean", dry), ("room", mined), ("both", np.minimum(dry, mined)))}
                rows.append(dict(voice=case["voice"], scene=case["scene"], rir=case["room"], mode=mode,
                                 start=case["start"], command_at=case["command_at"], pcm_sha256=identity,
                                 events=outputs, branch_peak_five_sum=peaks))
    assert not refs and len(rows) == 24
    save(RESULTS / f"{stem}-quiet.json", dict(baseline_sha256=sha(quiet), comparison_sha256=sha(comparison),
                                             plan_sha256=sha(PLAN), dry_archive_sha256=sha(archive), records=rows))


if __name__ == "__main__":
    main()
