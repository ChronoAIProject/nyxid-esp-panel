"""Check the Mac command clock estimate on known recording offsets and ambiguous playbacks."""
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import wave

import numpy as np
from scipy.signal import fftconvolve

import check_panel_music_transfer as transfer
import soundfile as sf

import check_connected_playback_codecs as study
import measure_connected_command_clock as clock

ch, tx = study.ch, study.tx
OUT = ch.RESULTS / "4090-c161-connected-command-clock.json"


def mr_capture(mic, channels):
    # Distinct other lanes catch accidental use of a playback reference or an unused mic.
    lanes = np.tile(np.array([0, 701, -337, 98][:channels], dtype="<i2"), (len(mic),1))
    lanes[:, 0] = mic
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setparams((channels, 2, 16000, 0, "NONE", "NONE"))
        wav.writeframes(lanes.tobytes())
    buf.seek(0)
    with wave.open(buf, "rb") as wav:
        result = clock.channel(wav, (2, 4))
    np.testing.assert_array_equal(result, mic)
    return result


def additional_rooms():
    songs = transfer.ch.read(transfer.music.fit.DEV)["records"][1:]
    rooms = sorted(transfer.music.ROOMS)
    rows = []
    for j, song in enumerate(songs):
        impulse, room, room_sha = tx.study.grid.room.rooms.read_room(f"h{rooms[2*j+1]:03}_")
        idx = (j % 12) * 5
        dry, command_at, _ = clock.track(idx)
        # Stretch the whole source on the capture clock to simulate independent audio clocks.
        ratio = 1.0002 if j % 2 else 0.9998
        t = np.arange(int(np.ceil(len(dry)*ratio))) / ratio
        changed = np.interp(t, np.arange(len(dry)), dry, right=0)
        wet = fftconvolve(changed, impulse)
        path = Path(song["path"])
        assert ch.sha(path) == song["sha256"]
        with sf.SoundFile(path) as f:
            assert f.samplerate == 16000 and f.channels == 1
            offset = 27017*(j+1) % (len(f)-len(wet))
            f.seek(offset)
            noise = f.read(len(wet))
        combined = wet + noise * (.60*np.linalg.norm(wet)/np.linalg.norm(noise))
        audio = np.rint(combined/max(abs(combined))*24000).astype("<i2")
        lead = (881*j) % 7000
        mic = np.pad(audio, (lead, 0))
        result = clock.locate(mic, dry, command_at)
        expected = lead + round(command_at*ratio) + int(np.abs(impulse).argmax())
        rows.append(dict(source_index=idx, room=room, room_sha256=room_sha, song_sha256=song["sha256"],
                         music_offset=offset, clock_ratio=ratio, recording_offset=lead, reference_sample=expected,
                         result=result, error_samples=result.get("estimated_command_sample", expected)-expected
                         if result["status"] == "matched" else None))
    assert len(rows) == 11
    return rows


def main():
    prior = ch.read(study.headroom.OUT)
    firmware_rows = {(r["profile"], r["previous_index"]): r for r in ch.read(study.OUT)["guest"]}
    calibration = study.playback.latencies()
    rows, negatives, summary, bank = [], [], defaultdict(Counter), {}
    for profile, role, s, pcm, _, old, _, _ in study.scenes(prior, calibration):
        if role != "both" or profile not in ("raw", "aac44_delayed", "sbc48_delayed"):
            continue
        idx = s["index"]
        if idx not in bank:
            bank[idx] = clock.track(idx)
        dry, start, _ = bank[idx]
        extra = (7103 * (idx + 1) + 171 * s["external_db"]) % 16000
        polarity = -1 if idx % 2 else 1
        channels = 4 if (idx + s["external_db"]//6) % 2 else 2
        mic = mr_capture(np.pad(polarity*pcm, (extra, 0)), channels)
        _, (impulse, _, _) = tx.diag.room(s["geometry"])
        dominant = int(np.abs(impulse).argmax())
        expected = extra + s["command_at"] + dominant
        board = old if profile == "raw" else firmware_rows[profile, old["index"]]
        assert board["mic_sha256"] == ch.digest(pcm)
        opened = board["opened"]
        capture_open = extra + opened[0] if opened else None
        result = clock.locate(mic, dry, start, capture_open)
        error = None
        if result["status"] == "matched":
            error = result["estimated_command_sample"] - expected
            if capture_open is not None:
                assert abs(result["estimated_lead_ms"] - (expected + error - capture_open)/16) < 1.e-9
        rows.append(dict(profile=profile, source_index=idx, external_db=s["external_db"], layout_channels=channels,
                         polarity=polarity, recording_offset=extra, room_peak=dominant, reference_sample=expected,
                         mic_sha256=ch.digest(mic), firmware_opened=opened, result=result, error_samples=error))
        summary[profile].update(trials=1, matched=int(error is not None),
                                within_10ms=int(error is not None and abs(error)<=160))
        if profile == "raw" and s["external_db"] == 0:
            other = (idx//15)*15 + ((idx%15 + 5)%15)
            wrong, wrong_at, _ = clock.track(other)
            variants = (("different_command", mic, wrong, wrong_at),
                        ("repeated_playback", np.r_[mic,np.zeros(4000,dtype="<i2"),mic], dry, start),
                        ("missing_command", mic[:extra+s["start"]], dry, start))
            for name, audio, template, at in variants:
                scored = clock.locate(audio, template, at)
                negatives.append(dict(index=idx, name=name, result=scored))
                summary[name].update(trials=1, matched=int(scored["status"]=="matched"))
    assert len(rows)==108 and len(negatives)==36
    additional = additional_rooms()
    tmp = OUT.with_suffix(".tmp")
    ch.save(tmp,dict(records=rows, controls=negatives, summary=summary, other_rooms=additional,
                     sources={p.name:ch.sha(p) for p in (clock.PACK, study.headroom.OUT, study.OUT, Path(clock.__file__),
                              Path(study.__file__), transfer.music.fit.DEV, Path(__file__))},
                     policy="Reference time is file command boundary plus calibrated codec delay, injected recording "
                            "offset and absolute-peak tap of the external room impulse. This approximates dominant "
                            "arrival. Use OPEN from the verified original or compressed P4 replay shifted by the injected recording offset. Alternate polarity and channel "
                            "layout. Compare a different command in the same voice, repeated and missing command."))
    tmp.replace(OUT)


if __name__ == "__main__":
    main()
