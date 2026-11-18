"""Isolate proposal cooldown on one consumed name prefix and compare daily negatives."""
import argparse
import hashlib
import shutil
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

import replay_c161_pcm_cues_v1 as matcher
import verify_c161_ambient_stream_v1 as live
from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding, cases
from select_c161_synthesized_cue_v2 import load_wav

REPORT = RESULTS / "4090-c161-proposal-cooldown-review-v1.json"
WITNESS = RESULTS / "hye-nyx-c161-chris-250ms-v3.wav"


def diagnostic_binding(root):
    # Isolate the single changed interval from the distributed sources.
    (root / "portable").mkdir()
    (root / "frontend_probe").mkdir()
    wrapper = Path("frontend_probe/c161_pcm_cue_matcher_v1.cpp")
    shutil.copyfile(matcher.ROOT / wrapper, root / wrapper)
    for path in ("c161_cue_suffix_v1.h", "c161_pcm_cue_coefficients_v1.h",
                 "c161_pcm_cue_matcher_v1.h", "c161_cue_suffix_coefficients_v1.h"):
        shutil.copyfile(matcher.ROOT / "portable" / path, root / "portable" / path)
    file = root / "portable/c161_pcm_cue_matcher_v1.h"
    old = file.read_text()
    assert old.count("step_ - last_ >= 4020") == 1
    # 825 ticks at 2 kHz outlast the latest suffix window (3168 + 128 ticks at 8 kHz).
    file.write_text(old.replace("step_ - last_ >= 4020", "step_ - last_ >= 825"))
    with patch.object(matcher, "ROOT", root), patch.object(matcher, "LIB", root / "short.so"):
        return matcher.binding(True)


def detect_short(lib, chunks, digest=None):
    state = lib.make_pcm_cue()
    if not state:
        raise MemoryError("PCM cue matcher allocation")
    try:
        position = -PRIME
        events = []
        for pcm in chain([np.zeros(PRIME, np.int16)], chunks):
            pcm = np.ascontiguousarray(pcm, dtype=np.int16)
            if digest is not None and position >= 0:
                digest.update(pcm.tobytes())
            # At most one return per sample, independent of proposal and suffix timing.
            samples = np.empty(len(pcm), np.int32)
            bits = np.empty(len(pcm), np.uint8)
            n = lib.pcm_cue_block(state, pcm.ctypes.data, len(pcm), samples.ctypes.data, bits.ctypes.data)
            events.extend(dict(t=position + int(t), cue_bits=int(b)) for t, b in zip(samples[:n], bits[:n], strict=True))
            position += len(pcm)
        return events
    finally:
        lib.destroy_pcm_cue(state)


def run(directory):
    assert sha(MODEL) == MODEL_SHA
    native = binding()
    with patch.object(matcher, "LIB", directory / "coarse.so"):
        coarse = matcher.binding()
    with patch.object(matcher, "LIB", directory / "confirmed.so"):
        frozen = matcher.binding(True)
    short = diagnostic_binding(directory)
    source = read(RESULTS / "4090-c161-suffix-cue-v3-source.json")
    v3 = load_wav(dict(path=RESULTS / "hye-nyx-c161-synthetic-cue-v3.wav", sha256=source["wav_sha256"]))
    frozen_daily = read(RESULTS / "4090-c161-confirmed-cue-daily-v1.json")["records"]
    daily_events = []
    for (case, pcm), old in zip(cases("daily"), frozen_daily, strict=True):
        assert case["key"] == old["key"] and hashlib.sha256(pcm.tobytes()).hexdigest() == old["pcm_sha256"]
        assert matcher.detect(frozen, [pcm]) == old["events"]
        events = detect_short(short, [pcm])
        if events:
            daily_events.append(dict(key=case["key"], text=case["text"], events=events))
        if case["key"] == "95":
            assert case["text"] == "Hey Chris, how was your day?"
            chris = pcm
    rows = []
    for gap in (0, 4000, 8000, 16000, 24000, 32160):
        for name, prefix in (("silence", np.zeros_like(chris)), ("Chris", chris)):
            pcm = np.r_[prefix, np.zeros(gap, np.int16), v3, np.zeros(8000, np.int16)]
            codes, clock, pair, _ = live.run_stream(native, pcm)
            base = times(decode(np.asarray(codes[0], np.uint8), np.asarray(clock)))
            confirmed = matcher.detect(frozen, (pcm,))
            candidate = detect_short(short, [pcm])
            rows.append(dict(history=name, gap=gap, start=len(prefix) + gap, end=len(prefix) + gap + len(v3),
                             pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(), original=base, pair=times(pair),
                             proposals=matcher.detect(coarse, (pcm,)), confirmed=confirmed, short_confirmed=candidate,
                             pair_with_cue=matcher.joined(times(pair), confirmed),
                             pair_with_short_cue=matcher.joined(times(pair), candidate)))
            if gap == 4000 and name == "Chris":
                sf.write(directory / WITNESS.name, pcm, 16000, subtype="PCM_16")
    candidate_wav = directory / WITNESS.name
    if WITNESS.exists():
        assert sha(candidate_wav) == sha(WITNESS)
    else:
        shutil.copyfile(candidate_wav, WITNESS)
    report = dict(model_sha256=MODEL_SHA, wav_sha256=source["wav_sha256"], witness_sha256=sha(WITNESS),
                  chris_pcm_sha256=hashlib.sha256(chris.tobytes()).hexdigest(), prime=PRIME,
                  coarse_ticks_original=4020, coarse_ticks_diagnostic=825, coarse_rate=2000,
                  frozen_daily_cases=len(frozen_daily), diagnostic_daily_events=daily_events, records=rows)
    if REPORT.exists():
        assert report == read(REPORT)
    else:
        save(REPORT, report)


def background(directory):
    lib = diagnostic_binding(directory)
    groups = {}
    for cohort in ("primed", "jenny", "melo"):
        original = read(RESULTS / f"4090-c161-confirmed-cue-background-{cohort}-v1.json")
        rows = []
        history = read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]
        sources = [r["source"] for r in history if r["phase"] == "background"]
        for source, control in zip(sources, original["records"], strict=True):
            path = Path(source.get("wav_path") or source["path"])
            digest = hashlib.sha256()
            with sf.SoundFile(path) as f:
                assert f.samplerate == 16000 and f.channels == 1 and f.subtype == "PCM_16"
                assert len(f) == source["samples"] == control["samples"]
                events = detect_short(lib, f.blocks(blocksize=96000, dtype="int16"), digest)
            assert digest.hexdigest() == source["pcm_sha256"] == control["pcm_sha256"]
            rows.append(dict(source_id=control["source_id"], pcm_sha256=digest.hexdigest(),
                             samples=control["samples"], original=control["events"], short_confirmed=events))
        groups[cohort] = rows
        print(cohort, len(rows), sum(len(r["short_confirmed"]) for r in rows), flush=True)
    output = RESULTS / "4090-c161-proposal-cooldown-background-v1.json"
    if output.exists():
        assert read(output) == groups
    else:
        save(output, groups)


def replay_consumed(directory):
    lib = diagnostic_binding(directory)
    groups = {}
    for name in ("cue", "official", "willow", "context", "generated", "joe", "mac"):
        control_file = RESULTS / f"4090-c161-confirmed-cue-{name}-v1.json"
        original = read(control_file)["records"]
        changes = []
        for control, (key, _, pcm, _, pair) in zip(original, matcher.cohort_data(name), strict=True):
            assert key == control["key"] and hashlib.sha256(pcm.tobytes()).hexdigest() == control["pcm_sha256"]
            short_events = detect_short(lib, [pcm])
            if short_events != control["events"]:
                changes.append(dict(key=key, pcm_sha256=control["pcm_sha256"], original=control["events"],
                                    short_confirmed=short_events, original_merged=control["pair_with_cue"],
                                    short_merged=matcher.joined(pair, short_events)))
        groups[name] = dict(baseline_report_sha256=sha(control_file), count=len(original), changes=changes)
    output = RESULTS / "4090-c161-proposal-cooldown-paired-v1.json"
    if output.exists():
        assert read(output) == groups
    else:
        save(output, groups)


def prospective_rooms(directory):
    source_file = RESULTS / "4090-c161-frozen-cue-prospective-v1.json"
    plan_file = RESULTS / "4090-c161-frozen-cue-prospective-plan-v1.json"
    source = read(source_file)
    assert sha(plan_file) == source["plan_sha256"]
    plan = read(plan_file)
    histories = [np.r_[load_wav(h["wav"]), np.zeros(h["gap"])] for h in plan["history"]]
    cues = {k: load_wav(dict(path=RESULTS / f"hye-nyx-c161-synthetic-cue-{k}.wav", sha256=v))
            for k, v in plan["waves"].items()}
    dry_rms = np.sqrt(np.mean(cues["v2"].astype(float) ** 2))
    rooms = {}
    for spec in plan["rooms"]:
        ir = np.zeros(spec["taps"][-1][0] + 1)
        for delay, gain in spec["taps"]:
            ir[delay] = gain
        rooms[spec["name"]] = ir, butter(2, spec["cutoff"], fs=16000, btype="bandpass", output="sos")
    with patch.object(matcher, "LIB", directory / "coarse.so"):
        coarse = matcher.binding()
    short = diagnostic_binding(directory)
    records = []
    for r in source["records"]:
        cue = cues[r["cue"]]
        rate = 1 + r["ppm"] / 1_000_000
        x = np.interp(np.arange(round(len(cue) / rate)) * rate, np.arange(len(cue)), cue)
        ir, sos = rooms[r["room"]]
        wet = sosfilt(sos, np.convolve(x, ir)) * 10 ** (r["gain_db"] / 20)
        full = np.r_[histories[r["history"]], wet, np.zeros(8000)]
        noise = np.random.default_rng(plan["seed"] + r["history"]).normal(size=len(full))
        noise = lfilter([1], [1, -.85], noise)
        full += noise * dry_rms / np.sqrt(np.mean(noise ** 2)) / 10 ** (r["nominal_snr"] / 20)
        pcm = np.rint(full).clip(-32768, 32767).astype("<i2")
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
        row = {key: r[key] for key in ("cue", "room", "pcm_sha256", "start", "end", "confirmed")}
        row.update(proposals=matcher.detect(coarse, [pcm]), short_confirmed=detect_short(short, [pcm]))
        records.append(row)
    report = dict(source_sha256=sha(source_file), records=records)
    output = RESULTS / "4090-c161-cue-prospective-room-stages-v1.json"
    if output.exists():
        assert read(output) == report
    else:
        save(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    suites = {"prefix": run, "background": background, "paired": replay_consumed, "rooms": prospective_rooms}
    parser.add_argument("suite", nargs="?", default="prefix", choices=suites)
    with TemporaryDirectory() as tmp:
        suites[parser.parse_args().suite](Path(tmp))
