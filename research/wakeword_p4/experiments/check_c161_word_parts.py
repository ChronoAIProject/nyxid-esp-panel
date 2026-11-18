"""Measure the labeled word window and test its acoustic dependence on each part of the request."""
import re

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from check_c161_room_word import detectors, events
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
import generate_c161_confuser_readout_v1 as confusers
import generate_c161_rhythm_pair_v1 as rhythm
import generate_c161_vowel_name_pair_v1 as vowels
from replay_c161_vowel_name_pair_v1 import binding
from replay_c161_word_window_v1 import score
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_mined_room_word import PREFIX
from train_c161_room_word import STEM as ROOM
from verify_c161_ambient_stream_v1 import run_stream

STEM = "4090-c161-word-parts"
FORWARD = RESULTS / "4090-c161-signed-forward-v1.json"


def boundaries(phones, durations):
    assert len(durations) == len(phones) + 2
    assert phones.startswith("hˈeɪ nˈ")
    edges = np.cumsum(durations, dtype=int) * 400
    second = phones.index("n")
    end = re.search(r"[,.?!]", phones).start()
    command = re.search(r"[^\sˌˈ]", phones[end + 1:])
    return dict(hey_start=int(edges[0]), name_start=int(edges[second]), word_end=int(edges[end]),
                command_start=int(edges[end + 1 + command.start()]) if command else None)


def training_marks():
    marks = {}
    for file in (vowels.SOURCE, confusers.SOURCE):
        for r in read(file)["records"]:
            if r.get("role", "fit") == "fit" and r["kind"] in ("nyx", "lowered"):
                marks[file.stem + "_" + r["key"]] = boundaries(r["phones"], r.get("duration") or r["durations"])
    for r in read(rhythm.META)["records"]:
        if r["positive"]:
            for label, variant in r["variants"].items():
                dur = np.asarray(r["natural_durations"])
                if label == "rhythm":
                    dur = dur + r["delta"]
                m = boundaries(r["phones"], dur)
                assert m["word_end"] == variant["end"]
                marks[rhythm.STEM + "_" + r["key"] + "_" + label] = m
    return marks


def mute(pcm, begin, end):
    # Taper inside the edited interval; every sample outside it is bit-identical.
    width = 64
    assert end - begin >= 2 * width
    out = pcm.copy()
    out[begin:end] = 0
    ramp = np.linspace(1, 0, width)
    out[begin:begin+width] = np.rint(pcm[begin:begin+width] * ramp)
    out[end-width:end] = np.rint(pcm[end-width:end] * ramp[::-1])
    return out


def main():
    candidate = f"{PREFIX}-hard"
    comparison = RESULTS / f"{candidate}-comparison-v1.json"
    training_file = RESULTS / f"{ROOM}-training-v1.json"
    anchors = training_marks()
    windows = []
    unaligned = []
    for r in read(training_file)["training"]:
        if not r["positive"]:
            continue
        if r["key"] not in anchors:
            unaligned.append(r["key"])
            continue
        m = anchors.pop(r["key"])
        assert m["word_end"] == r["end"]
        end = r["onset"] + r["end"]
        ticks = np.arange(((end - 480 + 479) // 480) * 480, end + 2881, 480) - r["onset"]
        windows.append(dict(key=r["key"], marks=m, ticks=ticks.tolist(),
                            nominal_hey_fully_visible=(ticks - 15360 <= m["hey_start"]).tolist(),
                            after_command=(ticks > m["command_start"]).tolist() if m["command_start"] is not None else []))
    assert not anchors
    straight = {r["meta"]["pcm_sha256"]: r for r in read(comparison)["records"] if r["cohort"] == "forward"}
    models = list(detectors(candidate))
    assert sha(MODEL) == MODEL_SHA
    native = binding()
    rows = []
    for r in read(FORWARD)["records"]:
        if r["target_id"] != "nyx":
            continue
        original = load_wav(r)
        m = boundaries(r["sources"]["nyx"]["phones"], r["sources"]["nyx"]["durations"])
        at = {k: r["start"] + v for k, v in m.items()}
        assert at["command_start"] == r["first_phone"]
        intervals = {"hey_muted": (r["start"], at["name_start"]),
                     "name_muted": (at["name_start"], at["word_end"]),
                     "wake_muted": (r["start"], at["command_start"]),
                     "command_muted": (at["command_start"], len(original))}
        waves = {"original": original, **{k: mute(original, a, b) for k, (a, b) in intervals.items()}}
        for label, pcm in waves.items():
            codes, ck, _, _ = run_stream(native, pcm)
            clock = np.array(ck)
            base = times(decode(np.asarray(codes[0], np.uint8), clock))
            (a, ct), (b, mt) = [score(it, pcm, inp, out) for it, inp, out in models]
            np.testing.assert_array_equal(ct, mt)
            np.testing.assert_array_equal(ct, clock)
            output = events(base, a, b, clock)
            if label == "original":
                assert output == straight[r["pcm_sha256"]]["events"]
            else:
                edit_begin = intervals[label][0]
                for k in output:
                    assert [t for t in output[k] if t <= edit_begin] == [t for t in waves_events[k] if t <= edit_begin]
            if label == "original":
                waves_events = output
            rows.append(dict(voice=r["voice"], scene=r["scene"], edit=label, marks=at, start=r["start"], end=r["end"],
                             input_sha256=r["pcm_sha256"], edited_sha256=digest(pcm), events=output))
    assert len(rows) == 60
    save(RESULTS / f"{STEM}.json", dict(forward_sha256=sha(FORWARD), comparison_sha256=sha(comparison),
         training_sha256=sha(training_file), aligned_training=windows, no_token_clock=unaligned,
         train_sources={str(p): sha(p) for p in (vowels.SOURCE, confusers.SOURCE, rhythm.META)}, records=rows))


if __name__ == "__main__":
    main()
