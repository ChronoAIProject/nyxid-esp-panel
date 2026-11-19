"""Time early and standard verifier windows on the archived calibrated playback residuals."""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
import torch

import check_calibrated_echo_names as names
import time_connected_verifiers as timing

ch, spoken = names.ch, names.spoken
STEM = "4090-c161-calibrated-echo-timing"
WORK = Path(".local/wakeword/c161-calibrated-echo-timing")
CHAR_BUILD = Path(".local/wakeword/c161-spelling-p4/build")
EARLY = 2880
STANDARD = 5760
CHAR = ch.RESULTS / (STEM + "-character.json")
FIVE = ch.RESULTS / (STEM + "-five.json")
OUT = ch.RESULTS / (STEM + ".json")


def signals(parent):
    """Reconstruct mic and residual using the archived filters, with byte-level cross-checks."""
    plan = ch.read(spoken.connected.PLAN)["records"]
    rows = iter(parent["records"])
    assert ch.sha(names.COEFF) == parent["coeff_sha256"]
    with np.load(names.COEFF, allow_pickle=False) as filters:
        for meta, self_mic, ref in spoken.extra_cases():
            if meta["mode"] != "self_only":
                continue
            h = filters[meta["bot"]]
            assert ch.digest(h) == parent["filters"][meta["bot"]]["float32_sha256"]
            prediction = names.estimate(ref, h)
            room, room_name, rir = ch.read_room(Path(meta["room"]).name[:5])
            assert (room_name, rir) == (meta["room"], meta["rir_sha256"])
            echo = fftconvolve(ref.astype(float), room)[:len(ref)] * meta["mic_echo_gain"]
            np.testing.assert_array_equal(np.rint(echo), self_mic)
            for source in (s for s in plan if s["voice"] == meta["voice"]):
                pcm = spoken.natural.replay.probe.dialogue.piper.load_wav(source)
                assert ch.digest(pcm) == source["pcm_sha256"]
                clip = pcm[source["start"]:]
                start = meta["cue_at"]
                length = source["end"] - source["start"]
                ratio = np.linalg.norm(echo[start:start+length]) / np.linalg.norm(clip[:length].astype(float))
                for db in (-6, 0, 6):
                    row = next(rows)
                    assert (row["voice"], row["role"], row["index"], row["db"], row["command_at"]) == (
                        source["voice"], source["target_id"], source["index"], db, start+length)
                    mix = echo.copy()
                    mix[start:start+len(clip)] += (10**(db/20)*ratio)*clip
                    mic = np.rint(mix).clip(-32768, 32767).astype("<i2")
                    residual = np.rint(mic - prediction.astype(float)).clip(-32768, 32767).astype("<i2")
                    residual[:parent["cutoff_samples"]] = 0
                    assert (ch.digest(mic), ch.digest(ref)) == (row["mic_sha256"], row["ref_sha256"])
                    assert ch.digest(residual) == row["stages"]["filtered"]["pcm_sha256"]
                    yield row, mic, residual
    assert next(rows, None) is None


def streams():
    net = spoken.natural.Detector()
    parent = ch.read(names.OUT)
    assert net.identity == parent["models"] and net.cutoff == 34
    assert spoken.natural.base.scoring.word.WAIT == STANDARD
    for index, (row, _, pcm) in enumerate(signals(parent)):
        old = row["stages"]["filtered"]
        proposals = old["proposals"]
        early = [t + EARLY for t in proposals if t + EARLY <= len(pcm)]
        trace = net.score(pcm, early)
        assert all(trace[k] == old[k] for k in ("pcm_sha256", "proposals", "events", "c161"))
        checks = trace["checks"]
        assert all(t >= parent["cutoff_samples"] for t in checks)
        accepted = {str(t): c["five"] for t, c in checks.items() if c["accepted"]}
        passed = [t for t, c in checks.items() if c["pass"]]
        yield "residual", index, pcm, [], {}, checks, accepted, timing.base.scoring.merged([], passed), {}


def collect():
    WORK.mkdir(parents=True, exist_ok=True)
    build = WORK / "build"
    if not build.exists():
        build.symlink_to((timing.WORK/"build").resolve(), target_is_directory=True)
    timing.collect(WORK, CHAR_BUILD, streams, [names.OUT, names.COEFF, spoken.connected.PLAN])


def block(t):
    return (t + names.HOP - 1) // names.HOP * names.HOP


def schedule(row, windows, delays, cutoff):
    old = row["stages"]["filtered"]
    direct = [block(t) for t in row["stages"]["raw"]["c161"]]
    direct.extend(block(t) for t in old["c161"] if t >= cutoff)
    accepted, passed, trace = set(), [], []
    done = 0
    jobs = sorted((t+d, t) for t in old["proposals"] for d in delays if t+d in windows)
    for end, proposal in jobs:
        if proposal in accepted:
            continue
        window = windows[end]
        ready = block(end)
        began = max(done, ready)
        done = began + (window["service_us"]*16 + 999)//1000
        if window["passes"]:
            accepted.add(proposal)
            passed.append(done)
        trace.append(dict(window=window["id"], proposal=proposal, end=end, ready=ready,
                          begin=began, done=done, passes=window["passes"]))
    events = timing.base.scoring.merged(sorted(set(direct)), passed)
    hits = [t for t in events if row["start"] <= t < row["end"]]
    return dict(events=events, jobs=trace, hit=bool(hits), before_command=any(t < row["command_at"] for t in hits))


def score():
    char, five, parent = ch.read(CHAR), ch.read(FIVE), ch.read(names.OUT)
    assert char["inputs"][names.OUT.name] == ch.sha(names.OUT)
    assert char["model_sha256"] == parent["models"]["character"] and five["model_sha256"] == parent["models"]["five"]
    assert five["qemu_sha256"] == char["qemu_sha256"]
    codes = {w["character_window"]: w for w in five["records"]}
    assert set(codes) == {i for i, w in enumerate(char["records"]) if w["capped"]}
    assert all(not w["differing_fields"] and not w["mismatched_bytes"] for w in char["records"])
    assert all(w["host"] == w["device"] for w in codes.values())
    records = []
    for i, (row, stream) in enumerate(zip(parent["records"], char["streams"], strict=True)):
        old = row["stages"]["filtered"]
        assert stream["index"] == i and stream["pcm_sha256"] == old["pcm_sha256"]
        windows = {}
        for j in stream["windows"]:
            w = char["records"][j]
            cost = w["invoke_us"] + w["decode_us"]
            if j in codes:
                cost += codes[j]["invoke_us"]
            windows[w["end"]] = dict(id=j, service_us=cost, passes=j in codes and codes[j]["device"][0] >= parent["models"]["cutoff"])
        original = schedule(row, windows, (STANDARD,), parent["cutoff_samples"])
        early = schedule(row, windows, (EARLY, STANDARD), parent["cutoff_samples"])
        # Rebuild the old (zero worker cost) delivered events, including independent raw and residual C161.
        no_cost = {t: dict(w, service_us=0) for t, w in windows.items()}
        assert schedule(row, no_cost, (STANDARD,), parent["cutoff_samples"])["events"] == old["available"]
        records.append(dict(ordinal=i, **{k:row[k] for k in ("index","voice","role","command","db","start","end",
                            "command_at","mic_sha256","ref_sha256")}, residual_sha256=old["pcm_sha256"],
                            proposals=old["proposals"], raw_panel=row["raw_panel"], standard=original, early=early))
    ch.save(OUT, dict(records=records, models=parent["models"], char_sha256=ch.sha(CHAR), five_sha256=ch.sha(FIVE),
            input_sha256=ch.sha(names.OUT), script_sha256=ch.sha(Path(__file__)), device=torch.cuda.get_device_name(),
            policy="Same calibrated mic and reference pairs, 16 kHz. Compare the standard 360 ms window with "
            "one extra 180 ms window using identical gates. Skip the fallback after an early acceptance. "
            "Round window availability to 256 samples, then charge measured per-window QEMU character Invoke, "
            "decode, and five-class Invoke if admitted. The two separate guests supply service time for a single "
            "simulated queue. End-to-end silicon costs, FIR, frontend and concurrent RTOS tasks are unmeasured."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "character", "five", "score"))
    action = parser.parse_args().action
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    if action == "collect":
        collect()
    elif action == "character":
        timing.spell.run(WORK/"character", 100, stem=CHAR.stem)
    elif action == "five":
        timing.measure_five(WORK, STEM)
    else:
        score()
