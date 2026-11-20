"""Prepare the eight physical and digital playback conditions for the Bella 072 false wake."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_bella_gain_false_wake as replay
import package_weak_gain_room_audio as room

daily, P = room.daily, room.P
NAME = "c161-bella-072-room-playback.zip"
REPORT = replay.OUT.with_name("4090-c161-bella-072-room-playback.json")


def rms_dbfs(signal):
    return float(20 * np.log10(np.sqrt(np.mean(signal.astype(float) ** 2)) / 32768))


def main():
    evidence = P.read(replay.OUT)
    old_pack = replay.OUT.with_suffix(".zip")
    assert P.sha(old_pack) == evidence["archive_sha256"]
    index = evidence["index"]
    row, = (r for r in P.read(daily.OUT)["records"] if r["index"] == index and r["role"] == "absent")
    grid, taps = daily.rooms()
    _, setup, render, ref, _, tracks = next(r for r in grid[row["db"]] if r[0]["file"] == row["file"])
    command = 2 * tracks["absent"].astype(np.int32)
    with ZipFile(old_pack) as saved:
        original_stereo = saved.read("mic-reference.wav")
        original_external = saved.read("speaker-quarter-gain.wav")
    combined, rate = sf.read(BytesIO(original_external), dtype="int32")
    assert rate == 16000 and np.all(combined % 16384 == 0)
    combined = combined // 16384
    interruption = combined - command
    start, end = evidence["interruption_start"], evidence["interruption_end"]
    assert not np.any(interruption[:start]) and not np.any(interruption[end:])
    assert not np.any(command[:evidence["command_at"]])
    silence = np.zeros_like(command)
    wet_silence = daily.board.bella.rear_filter(silence, taps)

    # Reuse the same pre-ADC room mix with the panel silenced for solo calibration.
    music, _, music_sha = daily.tx.study.grid.room.rooms.read_noise("music")
    v5, _ = daily.tx.study.panel_wave()
    panel, rate = sf.read(daily.study.mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000 and np.array_equal(ref[setup["panel_at"]:setup["panel_at"] + len(panel)], panel)
    (_, _, _), (rir, _, _) = daily.tx.diag.room(setup["geometry"])
    length = len(ref) - 4800 - len(rir) + 1
    solo = daily.study.mixed.room_mix(setup, {"external": (None, range(length))},
                                      np.zeros_like(panel), v5, music, music_sha)
    bed = solo(wet_silence) * setup["preamp_gain"]
    assert len(bed) == len(ref) and np.max(np.abs(bed)) < 32767
    panel_only, clips = render(wet_silence, include_music=False)
    assert clips == 0
    sources = {p.name: P.sha(p) for p in (Path(__file__), replay.OUT, old_pack, daily.OUT, room.MUSIC, room.LICENSE)}
    members, rows, levels = {}, [], {}
    filename = {(False, False): None, (True, False): "speaker/command.wav",
                (False, True): "speaker/interruption.wav", (True, True): "speaker/both.wav"}
    with TemporaryDirectory(dir=room.trial.WORK) as tmp:
        work = Path(tmp)
        with ZipFile(work / NAME, "w") as archive:
            def put(name, data):
                daily.matched.member(archive, name, data)
                members[name] = dict(bytes=len(data), sha256=sha256(data).hexdigest())

            put("music-source-LICENSE.txt", room.LICENSE.read_bytes())
            put("music-attribution.txt", (room.CREDIT + "\n").encode())
            put("panel.wav", room.wav(ref, "PCM_16"))
            put("music.wav", room.wav(np.rint(bed * 256).astype(np.int32) * 256, "PCM_24"))
            levels["music.wav"] = rms_dbfs(bed)
            levels["panel.wav"] = rms_dbfs(panel_only)
            for flags, dry in [((True, False), command), ((False, True), interruption), ((True, True), combined)]:
                assert np.max(np.abs(dry)) < 131072
                data = room.wav(dry * 16384, "PCM_24")
                if flags == (True, True):
                    assert data == original_external
                name = filename[flags]
                put(name, data)
                wet = daily.board.bella.rear_filter(dry, taps)
                levels[name] = rms_dbfs(solo(wet, include_music=False) * setup["preamp_gain"])

            for old in evidence["records"]:
                cmd, talk, music_on = old["command"], old["interruption"], old["music"]
                dry = (command if cmd else silence) + (interruption if talk else silence)
                mic, clips = render(daily.board.bella.rear_filter(dry, taps), include_music=music_on)
                assert clips == 0 and P.digest(mic) == old["mic_sha256"] and P.digest(ref) == old["reference_sha256"]
                stereo = room.wav(np.column_stack((mic, ref)), "PCM_16")
                if cmd and talk and music_on:
                    assert stereo == original_stereo
                digital = "digital/" + old["label"] + ".wav"
                put(digital, stereo)
                rows.append(dict(label=old["label"], speaker=filename[cmd, talk], music="music.wav" if music_on else None,
                                 panel="panel.wav", digital=digital, mic_sha256=old["mic_sha256"],
                                 reference_sha256=old["reference_sha256"], simulated_events=old["events"]))

            manifest = dict(index=index, rate=16000, full_scene_samples=len(ref), command=row["command"],
                            interruption=row["text"], command_at=evidence["command_at"], interruption_at=start,
                            interruption_end=end, panel_at=setup["panel_at"], records=rows, files=members,
                            solo_full_scene_mic_rms_dbfs=levels, sources=sources, attribution=room.CREDIT,
                            playback="Start all selected files at sample zero. Use the same external loudspeaker for "
                            "both voices, a second speaker for music, and the panel's own playout and reference. "
                            "A null source is muted. Calibrate each output using a solo board mic capture over "
                            "full_scene_samples, including silence and the acoustic tail. Hold gains and positions "
                            "fixed for the eight runs. For digital injection the stereo channels are M then R.")
            daily.matched.member(archive, "manifest.json", json.dumps(manifest, indent=2) + "\n")
        with ZipFile(work / NAME) as check:
            assert len(rows) == 8 and len(set(r["label"] for r in rows)) == 8
            assert check.testzip() is None and len(check.namelist()) == 16
        P.save(work / REPORT.name, dict(archive=NAME, bytes=(work / NAME).stat().st_size,
                                        archive_sha256=P.sha(work / NAME), **manifest))
        (work / NAME).replace(room.trial.WORK / NAME)
        (work / REPORT.name).replace(REPORT)


if __name__ == "__main__":
    main()
