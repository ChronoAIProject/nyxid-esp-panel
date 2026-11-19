"""Exercise source clock error and independent reference drift on the frozen handoff cue."""
import argparse
from fractions import Fraction
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import scipy
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import handoff_panel_phase as phase
from code_gain_guard import activity_pulses
from replay_c161_pcm_cues_v1 import detect
from self_guard import expected

handoff, ch, listener = phase.handoff, phase.ch, phase.listener
fixed = handoff.probe.fixed
RATES = (0, -100, 100, -1000, 1000, -5000, 5000)
STEM = ch.RESULTS / "4090-c161-handoff-clock-rates"
GRID = STEM.with_name(STEM.name + "-grid.json")
BASELINE = ch.RESULTS / f"{fixed.STEM}-p4.json"
DRIFT_BASELINE = handoff.probe.BASE.with_suffix(".json")
FIELDS = ("mic_sha256", "reference_sha256", "c161", "fixed", "reference_events",
          "prefix_reference", "qualified_prefix", "allowed", "opened")


def project(pcm, ratio):
    """Render at output/input sample-count ratio, aligned at the first sample."""
    if ratio == 1:
        return pcm
    out = np.rint(resample_poly(pcm.astype(float), ratio.numerator, ratio.denominator))
    assert np.all((out >= -32768) & (out <= 32767))
    return out.astype("<i2")


def materials():
    prior = ch.read(BASELINE)
    assert prior["playback_sha256"] == ch.sha(handoff.COMMANDS)
    old = {(r["voice"], r["carrier"], r["room"]): r for r in prior["records"]}
    with ZipFile(handoff.COMMANDS) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        assert manifest["cue_sha256"] == ch.sha(fixed.CUE)
        cue, rate = sf.read(fixed.CUE, dtype="int16")
        assert rate == 16000 and len(cue) == manifest["cue_samples"]
        tracks = {}
        for item in manifest["records"]:
            data = pack.read(item["file"])
            track, rate = sf.read(io.BytesIO(data), dtype="int16")
            assert rate == 16000 and ch.digest(track) == item["pcm_sha256"]
            np.testing.assert_array_equal(track[:len(cue)], cue)
            tracks[item["voice"], item["carrier"]] = track
    plan = ch.read(ch.PLAN)["records"]
    for rir, kind in (("h012_", "air_conditioner"), ("h004_", "music")):
        h, room, rh = ch.read_room(rir)
        noise, _, nh = ch.read_noise(kind)
        for s in plan:
            if (s["room"], s["target_id"]) != (room, "coded"):
                continue
            assert (rh, nh) == (s["rir_sha256"], s["noise_sha256"])
            key = (s["voice"], s["carrier"])
            witness = old.pop((*key, room))
            start = witness["cue_at"]
            assert s["end"] == start + len(cue)
            track = tracks[key]
            for ppm in RATES:
                ratio = Fraction(1000000, 1000000 + ppm)
                command = start + round(len(cue) * ratio)
                for mode in ("fixed", "cue_muted"):
                    original = track if mode == "fixed" else np.r_[np.zeros_like(cue), track[len(cue):]]
                    source = project(original, ratio)
                    wet = fftconvolve(source.astype(float), h)
                    signal = noise[s["noise_offset"]:s["noise_offset"]+s["samples"]] * s["noise_gain"]
                    assert start + len(wet) <= len(signal)
                    signal[start:start+len(wet)] += wet
                    q = np.rint(s["common_gain"] * signal)
                    mic = q.clip(-32768, 32767).astype("<i2")
                    if mode == "fixed" and ppm == 0:
                        assert ch.digest(mic) == witness["mic_sha256"]
                    ref = np.zeros_like(mic)
                    if mode == "fixed":
                        ref[start:start+len(source)] = source
                    meta = dict(room=room, voice=key[0], carrier=key[1], cue_at=start,
                                command_at=command, mode=mode, player_ppm=ppm)
                    yield meta, mic, ref, int(np.count_nonzero(q != mic)), witness
    assert not old


def independent_reference():
    archive = ch.read(DRIFT_BASELINE)["records"]
    for src, mic, original, _ in handoff.probe.shower.streams():
        if not (src["mode"] == "voip-12" and src["noisy"] and src["player"] == "self"
                and Path(src["room"]).name.startswith("h024_")):
            continue
        witness, = (r for r in archive if r["group"] == "rooms" and r["source"] == src)
        assert ch.digest(mic) == witness["mic_sha256"] and ch.digest(original) == witness["reference_sha256"]
        a, b = src["cue_at"], src["command_at"]
        assert not original[:a].any() and not original[b:].any()
        for ppm, ages in ((0, (0,)), (-100, (0, 600, 1120, 1280, 1500)), (100, (0, 600, 1120, 1280, 1500))):
            # Positive error means additional reference samples per elapsed mic second.
            ratio = Fraction(1000000 + ppm, 1000000)
            cue = project(original[a:b], ratio)
            for age in ages:
                offset = round(Fraction(age * 16000 * ppm, 1000000))
                ref = np.zeros_like(mic)
                assert a + offset >= 0 and a + offset + len(cue) <= len(ref)
                ref[a+offset:a+offset+len(cue)] = cue
                yield dict(src, reference_ppm=ppm, seconds_since_sync=age, offset_samples=offset), mic, ref, witness


def guarded(mic_routes, full, pair, ref):
    fixed_events = [r["t"] for r in mic_routes["fixed"]]
    refs = [r["t"] for r in detect(full, [ref])]
    prefixes = [r["t"] for r in detect(pair, [ref])]
    signed = [t for t in prefixes if handoff.probe.qualify(ref, t)]
    waiting = sorted(set(prefixes) - set(signed))
    allowed = sorted(expected(fixed_events + mic_routes["c161"], refs, activity_pulses(ref), signed, waiting))
    return dict(mic_sha256=mic_routes["pcm_sha256"], reference_sha256=ch.digest(ref),
                c161=mic_routes["c161"], fixed=fixed_events, reference_events=refs, prefix_reference=prefixes,
                qualified_prefix=signed, allowed=allowed, opened=[(allowed[0]+255)//256*256] if allowed else [])


def grid():
    native = ch.c161_binding()
    rows, drift = [], []
    with TemporaryDirectory() as tmp:
        full = ch.binding(Path(tmp)/"v5.so", coded="v5")
        pair = ch.binding(Path(tmp)/"pair.so", coded="aligned")
        for meta, mic, ref, clips, witness in materials():
            routes = ch.routes(full, native, mic)
            external = guarded(routes, full, pair, np.zeros_like(ref))
            row = dict(source=meta, clips=clips, external=external)
            if meta["mode"] == "fixed":
                row["self"] = guarded(routes, full, pair, ref)
                if meta["player_ppm"] == 0:
                    assert external["opened"] == [witness["external"]["delivered"]]
                    assert external["fixed"] == witness["external"]["v5"] and external["c161"] == witness["external"]["c161"]
            rows.append(row)
        for meta, mic, ref, witness in independent_reference():
            result = guarded(ch.routes(full, native, mic), full, pair, ref)
            if meta["reference_ppm"] == 0:
                assert all(result[k] == witness[k] for k in FIELDS)
            drift.append(dict(source=meta, result=result))
    assert len(rows) == 224 and len(drift) == 11
    ch.save(GRID, dict(records=rows, drift=drift, source_rates_ppm=RATES, scipy=scipy.__version__,
        cue_sha256=ch.sha(fixed.CUE), pack_sha256=ch.sha(handoff.COMMANDS), baseline_sha256=ch.sha(BASELINE),
        model_sha256=listener.C161, rooms_plan_sha256=ch.sha(ch.PLAN), drift_baseline_sha256=ch.sha(DRIFT_BASELINE),
        script_sha256=ch.sha(Path(__file__)), policy="Faster source for positive player ppm. Bandlimited polyphase "
        "resampling of the entire dry file before room convolution, with the original background on the receiver "
        "clock. Round the command boundary to the nearest output sample. External reference is silent. Self uses "
        "the entire projected file on the receiver clock. Muted controls keep the command. The separate bathroom "
        "probe holds the mic fixed and projects the Opus cue reference onto a clock with a specified accumulated "
        "offset at cue start. No long recording or physical clock measurement."))


def chosen(meta):
    # Fix two room/voice/command combinations before scoring any clock setting.
    return (meta["mode"] == "fixed" and meta["player_ppm"] in (0, -5000, 5000)
            and (Path(meta["room"]).name[:4], meta["voice"], meta["carrier"]) in (
                ("h012", "af_bella", ch.TEXTS[0]), ("h004", "bm_fable", ch.TEXTS[1])))


def p4():
    grid_data = ch.read(GRID)
    assert grid_data["script_sha256"] == ch.sha(Path(__file__))
    history = ch.read(phase.HISTORY)
    fw = handoff.test.BUILD/"nyx_mac_listener.bin"
    codec = fixed.uplink.CODEC/"nyx_first_command.bin"
    model = listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model) == listener.C161 and fw.read_bytes().count(model.read_bytes()) == 1
    assert (ch.sha(fw), ch.sha(codec), ch.sha(listener.QEMU)) == (
        history["firmware_sha256"], history["codec_sha256"], history["qemu_sha256"])
    reader, identity = ch.load_reader()
    assert identity == history["asr"]
    decoder = fixed.uplink.decoder_api()
    records = []
    archive = STEM.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        full = ch.binding(work/"v5.so", coded="v5")
        pair = ch.binding(work/"pair.so", coded="aligned")

        def replay(meta, mic, ref, host, player):
            index = len(records)
            board = phase.probe(work, meta, mic, ref, full, pair, index)
            assert all(board[k] == host[k] for k in FIELDS)
            if player == "external" and board["opened"]:
                board["uplink"] = handoff.bench.fifo(work, logs, index, mic, ref, board, reader, decoder, True, True)
            logs.writestr(f"{index:02}-listener.log", "\n".join(board["uart_events"])+"\n")
            records.append(dict(player=player, **board))
            print(index, player, board["opened"], flush=True)

        for (meta, mic, ref, clips, _), row in zip(materials(), grid_data["records"], strict=True):
            assert row["source"] == meta and row["clips"] == clips
            if chosen(meta):
                replay(meta, mic, np.zeros_like(ref), row["external"], "external")
                replay(meta, mic, ref, row["self"], "self")
        for (meta, mic, ref, _), row in zip(independent_reference(), grid_data["drift"], strict=True):
            assert row["source"] == meta
            replay(meta, mic, ref, row["result"], "independent_reference")
    assert len(records) == 23
    archive.with_suffix(".tmp").replace(archive)
    ch.save(STEM.with_suffix(".json"), dict(records=records, asr=identity, grid_sha256=ch.sha(GRID),
        archive_sha256=ch.sha(archive), firmware_sha256=ch.sha(fw), codec_sha256=ch.sha(codec),
        qemu_sha256=ch.sha(listener.QEMU), model_sha256=ch.sha(model), script_sha256=ch.sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("grid", "p4"))
    {"grid": grid, "p4": p4}[parser.parse_args().action]()
