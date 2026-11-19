"""Package the actual external-only WAV and digital speaker reference for paired Mac playback."""
from hashlib import sha256
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

import check_fixed_cue_over_bot as fixed
import check_fixed_cue_playback_stop as stop

spoken, ch, uplink = fixed.spoken, fixed.ch, fixed.uplink
PACK = ch.RESULTS / "4090-c161-fixed-cue-over-spoken-bot-playback.zip"


def main():
    old = ch.read(fixed.OUT)
    cases = {(r["source"]["voice"], r["source"]["external_to_echo_db"], r["source"]["mode"]): r
             for r in old["records"] if r["source"].get("role") == "fixed-v5"}
    files, records = {}, []
    with ZipFile(PACK.with_suffix(".tmp"), "w", ZIP_DEFLATED) as z:
        for s, mic, ref in spoken.extra_cases():
            if s.get("role") != "fixed-v5":
                continue
            voice, db = s["voice"], s["external_to_echo_db"]
            r = cases[voice, db, s["mode"]]
            assert ch.digest(mic) == r["mic_sha256"] and ch.digest(ref) == r["ref_sha256"]
            external = f"external/{voice}_{db:+d}dB.wav"
            panel = "panel/" + s["bot"] + ".wav"
            if s["mode"] == "external_only":
                assert not ref.any() and not mic[:s["cue_at"]].any()
                files[external] = uplink.write_member(z, external, mic)
                continue
            if panel not in files:
                files[panel] = uplink.write_member(z, panel, ref)
            assert files[panel]["pcm_sha256"] == r["ref_sha256"]
            assert files[external]["pcm_sha256"] == cases[voice, db, "external_only"]["mic_sha256"]
            records.append(dict(external=external, panel=panel, external_to_echo_db_in_simulation=db,
                                voice=voice, wake_at=s["cue_at"], command_at=s["command_at"],
                                expected_external_command=s["carrier"], panel_says=s["bot_text"],
                                reference_room=s["room"], qemu_open=r["open"][0], original_case=r["index"]))
        assert len(files) == 14 and len(records) == 12
        manifest = dict(sample_rate=16000, encoding="mono PCM16", sample_origin=0, cue="fixed-v5",
                        model_sha256=spoken.natural.MODEL_SHA, raw_cue_sha256=ch.sha(stop.audit.rooms.fixed.CUE),
                        inputs_and_results={p.name: ch.sha(p) for p in (fixed.OUT, stop.OUT)},
                        files=files, pairs=records, playback="Start external and panel WAVs on the same clock. "
                        "Route external to a separate speaker and panel to its own playback path. "
                        "For external-only or panel-only mute the other channel. The dB labels describe "
                        "the simulated mic ratio during the wake cue; measure actual levels at the device. "
                        "For the interrupted arm fade only the panel on the device's observed OPEN. "
                        "Simulation used a 10 ms fade and zero or 100 ms of queued playback.")
        z.writestr(ZipInfo("manifest.json"), json.dumps(manifest, indent=2) + "\n", compress_type=ZIP_DEFLATED)
    with ZipFile(PACK.with_suffix(".tmp")) as z:
        assert set(z.namelist()) == {*files, "manifest.json"} and z.testzip() is None
        for name, meta in files.items():
            blob = z.read(name)
            pcm, rate = sf.read(io.BytesIO(blob), dtype="int16")
            assert rate == 16000 and pcm.ndim == 1 and pcm.size == meta["samples"]
            assert len(blob) == meta["bytes"] and sha256(blob).hexdigest() == meta["wav_sha256"]
            assert ch.digest(pcm) == meta["pcm_sha256"]
    PACK.with_suffix(".tmp").replace(PACK)


if __name__ == "__main__":
    main()
