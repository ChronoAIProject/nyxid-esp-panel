"""Sweep panel speech phase, then replay the loudest overlap during the wake codes on P4."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import handoff_barge_in as handoff
from code_gain_guard import activity_pulses
from replay_c161_pcm_cues_v1 import detect
from self_guard import expected

ch, listener = handoff.fixed.ch, handoff.listener
BASE = listener.RESULTS / "4090-c161-handoff-panel-phase"
GRID = BASE.with_name(BASE.name + "-grid.json")
HISTORY = handoff.BASE.with_suffix(".json")


def sources(player):
    # Rank the *digital speaker*, independent of recognizer, room or microphone.
    info = listener.read(handoff.probe.fixed.SOURCE_INFO)
    positions = info["code_positions"]
    length = info["code_length"]
    assert len(positions) == 3 and info["wav_sha256"] == listener.sha(handoff.probe.fixed.CUE)
    silence_plan = { (r["room"], r["voice"], r["carrier"]): r for r in listener.read(ch.PLAN)["records"]
                     if r["target_id"] == "silence" }
    history = listener.read(HISTORY)["records"]
    for (meta, dry, original, response), old in zip(handoff.scenes(), history, strict=True):
        assert meta == old["source"]
        quiet = silence_plan[(meta["room"], meta["voice"], meta["carrier"])]
        assert listener.sha(Path(quiet["path"])) == quiet["sha256"]
        control, rate = sf.read(quiet["path"], dtype="int16")
        assert rate == 16000 and control.shape == dry.shape and listener.digest(control) == quiet["pcm_sha256"]
        noise = np.random.default_rng(18221).standard_normal(len(dry))
        phases = []
        for j in range(8):
            offset = len(player)*j//8
            ref = np.resize(np.roll(player, -offset), len(dry))
            if j == 0:
                np.testing.assert_array_equal(ref, original)
            windows = [ref[meta["cue_at"]+p:meta["cue_at"]+p+length].astype(float) for p in positions]
            phases.append(dict(offset=offset, code_rms=float(np.sqrt(np.mean(np.square(windows))))))
        yield meta, dry, control, response, noise, phases, old


def probe(work, meta, mic, adc, full, pair, index):
    return handoff.probe.score(work, meta, mic, adc, full, index, build=handoff.test.BUILD,
                prefix=pair, qualify=handoff.probe.qualify, wait_for_third=True,
                frame_activity=True, qualify_capture_frame=True)


def grid():
    native = ch.c161_binding()
    player = handoff.speaker()
    rows = []
    with TemporaryDirectory() as tmpdir:
        full = ch.binding(Path(tmpdir)/"full.so", coded="v5")
        pair = ch.binding(Path(tmpdir)/"pair.so", coded="aligned")
        for meta, dry, control, response, noise, phases, old in sources(player):
            for phase in phases:
                ref = np.resize(np.roll(player, -phase["offset"]), len(dry))
                scene = dict(source=meta, **phase)
                for role, before_echo in (("fixed", dry), ("silence", control)):
                    mic, adc, clips = handoff.render(before_echo, ref, response, meta["echo_gain"], noise)
                    routes = ch.routes(full, native, mic)
                    fixed = [r["t"] for r in routes["fixed"]]
                    ref_full = [r["t"] for r in detect(full, [adc])]
                    ref_pair = [r["t"] for r in detect(pair, [adc])]
                    signed = [t for t in ref_pair if handoff.probe.qualify(adc, t)]
                    wait = sorted(set(ref_pair)-set(signed))
                    allowed = sorted(expected(fixed+routes["c161"], ref_full, activity_pulses(adc), signed, wait))
                    opened = [(allowed[0]+255)//256*256] if allowed else []
                    result = dict(mic_sha256=listener.digest(mic), reference_sha256=listener.digest(adc),
                                  clipped_samples=clips, fixed=fixed, c161=routes["c161"],
                                  reference_full=ref_full, reference_pair=ref_pair, opened=opened)
                    if phase["offset"] == 0 and role == "fixed":
                        assert all(result[k] == old["continue"][k] for k in
                                   ("mic_sha256", "reference_sha256", "fixed", "c161", "opened"))
                    scene[role] = result
                rows.append(scene)
            print(meta["voice"], meta["carrier"], meta["room"], flush=True)
    assert len(rows) == 128
    listener.save(GRID, dict(records=rows, code_sha256=listener.sha(Path(__file__)),
        parent_sha256=listener.sha(HISTORY), silence_plan_sha256=listener.sha(ch.PLAN),
        player_pcm_sha256=listener.digest(player), model_sha256=listener.C161,
        policy="Eight equally spaced circular offsets of the same admitted 7.3-second digital panel speech. "
               "Hold room, echo gain, command, ambient and reference noise fixed. Zero-cue controls retain the "
               "entire external command. First delivery from primed raw C161 and fixed routes, reference "
               "protection and native 256-sample polling. Select highest combined RMS in the three source-code "
               "windows for P4. Reused conditions, no new background hours."))


def p4():
    data = listener.read(GRID)
    history = listener.read(HISTORY)
    assert data["parent_sha256"] == listener.sha(HISTORY)
    reader, identity = listener.load_reader()
    assert history["asr"] == identity
    fw = handoff.test.BUILD/"nyx_mac_listener.bin"
    assert listener.sha(fw) == history["firmware_sha256"] and listener.sha(listener.QEMU) == history["qemu_sha256"]
    codec = handoff.probe.uplink.CODEC/"nyx_first_command.bin"
    assert listener.sha(codec) == history["codec_sha256"]
    decoder = handoff.probe.uplink.decoder_api()
    archive = BASE.with_suffix(".zip")
    tmp_zip = BASE.with_suffix(".tmp")
    player = handoff.speaker()
    rows = []
    with TemporaryDirectory() as tmpdir, ZipFile(tmp_zip, "w", ZIP_DEFLATED) as logs:
        work = Path(tmpdir)
        full = ch.binding(work/"full.so", coded="v5")
        pair = ch.binding(work/"pair.so", coded="aligned")
        for j, (source, dry, control, response, noise, phases, _) in enumerate(sources(player)):
            peak = max(range(len(phases)), key=lambda i: phases[i]["code_rms"])
            witness = data["records"][8*j+peak]
            phase = phases[peak]
            assert witness["source"] == source and witness["offset"] == phase["offset"]
            player_pcm = np.resize(np.roll(player, -phase["offset"]), len(dry))
            row = dict(source=source, **phase)
            for arm in ("silence", "continue", "interrupt"):
                speaker = player_pcm.copy()
                if arm == "interrupt":
                    if not row["continue"]["opened"]:
                        break
                    stop = row["continue"]["opened"][0]
                    fade = handoff.FADE
                    speaker[stop:stop+fade] = np.rint(speaker[stop:stop+fade]*np.arange(fade-1,-1,-1)/fade)
                    speaker[stop+fade:] = 0
                mic, adc, clips = handoff.render(control if arm == "silence" else dry, speaker, response,
                                                  source["echo_gain"], noise)
                index = 3*j + ("silence", "continue", "interrupt").index(arm)
                result = probe(work, source, mic, adc, full, pair, index)
                if arm == "continue":
                    original_mic, original_adc = mic, adc
                if arm == "interrupt":
                    np.testing.assert_array_equal(original_mic[:stop], mic[:stop])
                    np.testing.assert_array_equal(original_adc[:stop], adc[:stop])
                    assert result["opened"] == row["continue"]["opened"]
                else:
                    host = witness["silence" if arm == "silence" else "fixed"]
                    assert all(result[k] == host[k] for k in ("mic_sha256","reference_sha256","fixed","c161","opened"))
                if result["opened"]:
                    result["uplink"] = handoff.bench.fifo(work, logs, index, mic, adc, result, reader, decoder, True, True)
                result.update(clipped_samples=clips, speaker_pcm_sha256=listener.digest(speaker))
                logs.writestr(f"{index:02}-listener.log", "\n".join(result["uart_events"])+"\n")
                row[arm] = result
                print(index, arm, result["opened"], flush=True)
            rows.append(row)
    assert len(rows) == 16
    tmp_zip.replace(archive)
    listener.save(BASE.with_suffix(".json"), dict(records=rows, grid_sha256=listener.sha(GRID),
        archive_sha256=listener.sha(archive), code_sha256=listener.sha(Path(__file__)),
        firmware_sha256=listener.sha(fw), codec_sha256=listener.sha(codec), asr=identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("grid", "p4"))
    {"grid": grid, "p4": p4}[parser.parse_args().action]()
