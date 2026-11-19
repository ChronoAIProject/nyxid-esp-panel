"""Resample and encode the handed-off cue plus command before the two-room P4 replay."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import av
import numpy as np
from scipy.signal import correlate, fftconvolve, resample_poly

import coded_fixed_transport as transport
import handoff_barge_in as handoff
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

ch = transport.ch
OUT = RESULTS / "4090-c161-handoff-playback-codecs"
PROFILES = {
    "raw": (16000, None, 0),
    "pcm44": (44100, None, 0), "aac44": (44100, "aac", 128000),
    "pcm48": (48000, None, 0), "sbc48": (48000, "sbc", 328000),
}


def encode_decode(pcm, profile):
    rate, codec, bitrate = PROFILES[profile]
    if profile == "raw":
        return pcm.copy(), {}
    data = resample_poly(pcm.astype(float), rate, 16000)
    info = dict(samples_before_codec=len(data))
    if codec:
        enc = av.CodecContext.create(codec, "w")
        enc.sample_rate, enc.layout, enc.bit_rate = rate, "stereo", bitrate
        enc.format = enc.codec.audio_formats[0]
        enc.thread_count = 1
        enc.open()
        if enc.format.name == "fltp":
            stereo = np.tile((data / 32768).astype(np.float32), (2, 1))
        else:
            assert enc.format.name == "s16"
            stereo = np.repeat(np.rint(data).clip(-32768, 32767).astype("<i2"), 2).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(stereo, format=enc.format.name, layout="stereo")
        frame.sample_rate = rate
        packets = enc.encode(frame) + enc.encode(None)
        decoder = av.CodecContext.create(codec, "r")
        decoder.sample_rate, decoder.layout = rate, "stereo"
        decoder.thread_count = 1
        decoder.extradata = enc.extradata
        frames = []
        for packet in packets:
            # Decode bare payloads, without container-side trimming of encoder delay.
            frames.extend(decoder.decode(av.Packet(bytes(packet))))
        frames.extend(decoder.decode(None))
        pieces = []
        for decoded in frames:
            assert decoded.sample_rate == rate and len(decoded.layout.channels) == 2
            data = decoded.to_ndarray()
            stereo = data if decoded.format.is_planar else data.reshape(-1, 2).T
            mono = stereo.astype(float).mean(axis=0)
            if data.dtype.kind == "f":
                mono *= 32768
            else:
                assert data.dtype == np.int16
            pieces.append(mono)
        data = np.concatenate(pieces)
        wire = b"".join(bytes(packet) for packet in packets)
        info.update(packets=len(packets), bytes=len(wire), payload_sha256=sha256(wire).hexdigest())
    out = np.rint(resample_poly(data, 16000, rate))
    info.update(output_clips=int(np.sum(np.abs(out.clip(-32768, 32767) - out) > 0)))
    return out.clip(-32768, 32767).astype("<i2"), info


def latencies():
    # Independent seeded calibration stimulus measures bulk lag on the 16 kHz clock.
    noise = np.random.default_rng(78251).integers(-6000, 6001, 8192, dtype=np.int16)
    noise = np.rint(np.convolve(noise, [0.25, 0.5, 0.25], mode="same")).astype("<i2")
    probe = np.pad(noise, (2048, 4096))
    reports = {}
    for name in PROFILES:
        out, info = encode_decode(probe, name)
        lo = int(correlate(out.astype(float), noise.astype(float), mode="valid").argmax())
        lag = lo - 2048
        assert 0 <= lag < 1024
        reports[name] = dict(lag=lag, correlation=float(np.corrcoef(out[lo:lo+len(noise)], noise)[0, 1]),
                             input_sha256=ch.digest(probe), output_sha256=ch.digest(out), **info)
        assert reports[name]["correlation"] > .85
    return reports


def main():
    fixed = handoff.probe.fixed
    source_info = read(fixed.SOURCE_INFO)
    history = read(RESULTS / f"{fixed.STEM}-p4.json")
    assert history["playback_sha256"] == sha(handoff.COMMANDS)
    assert source_info["wav_sha256"] == sha(fixed.CUE)
    pin = {(r["room"], r["voice"], r["carrier"]): r for r in history["records"]}
    with ZipFile(handoff.COMMANDS) as source:
        import io
        import json
        import soundfile as sf
        files = {}
        for item in json.loads(source.read("manifest.json"))["records"]:
            wave = source.read(item["file"])
            assert sha256(wave).hexdigest() == item["wav_sha256"]
            pcm, rate = sf.read(io.BytesIO(wave), dtype="int16")
            assert rate == 16000 and ch.digest(pcm) == item["pcm_sha256"]
            files[item["voice"], item["carrier"]] = pcm
    delays = latencies()
    signals = {}
    for rir, kind in (("h012_", "air_conditioner"), ("h004_", "music")):
        impulse, room, rh = ch.read_room(rir)
        noise, _, nh = ch.read_noise(kind)
        signals[room] = impulse, noise, rh, nh
    cache, tracks, rows = {}, [], []
    native = ch.c161_binding()
    uplink = fixed.uplink
    reader, identity = ch.load_reader()
    assert identity == history["asr_files"]
    firmware = uplink.firmware()
    assert all(history[k] == v for k, v in firmware.items())
    decoder = uplink.decoder_api()
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        detector = ch.binding(work / "fixed.so", coded="v5")
        for _, s, version, baseline, cue, cmd in transport.materials():
            if version != "advance":
                continue
            track = files[s["voice"], s["carrier"]]
            np.testing.assert_array_equal(track, np.r_[cue, cmd])
            impulse, noise, rh, nh = signals[s["room"]]
            assert (rh, nh) == (s["rir_sha256"], s["noise_sha256"])
            bg = noise[s["noise_offset"]:s["noise_offset"]+len(baseline)] * s["noise_gain"]
            old = pin.pop((s["room"], s["voice"], s["carrier"]))
            for mode in ("fixed", "silence"):
                original = track if mode == "fixed" else np.r_[np.zeros_like(cue), cmd]
                for profile in PROFILES:
                    key = (ch.digest(original), profile)
                    if key not in cache:
                        # Flush the resamplers and codec tails identically for all playback files.
                        source, info = encode_decode(np.pad(original, (0, 2048)), profile)
                        lag = delays[profile]["lag"]
                        aligned = source[lag:lag+len(original)]
                        assert len(aligned) == len(original)
                        cache[key] = (len(tracks), aligned)
                        tracks.append(dict(profile=profile, mode=mode, input_sha256=key[0],
                                           aligned_sha256=ch.digest(aligned), **info))
                    track_id, pcm = cache[key]
                    wet = fftconvolve(pcm.astype(float), impulse)
                    signal = bg.copy()
                    start, at = s["start"], s["end"]
                    signal[start:start+len(wet)] += wet
                    q = np.rint(s["common_gain"] * signal)
                    mic = q.clip(-32768, 32767).astype("<i2")
                    scored = ch.routes(detector, native, mic)
                    clocks = scored["merged"]
                    opened = (clocks[0]+255)//256*256 if clocks else None
                    row = dict(mode=mode, profile=profile, track=track_id, room=s["room"],
                               voice=s["voice"], carrier=s["carrier"], start=start, command_at=at,
                               mic_sha256=ch.digest(mic), clips=int(np.count_nonzero(q != mic)),
                               c161=scored["c161"], fixed=[event["t"] for event in scored["fixed"]], opened=opened)
                    if mode == "fixed" and profile == "raw":
                        np.testing.assert_array_equal(mic, baseline)
                        assert row["mic_sha256"] == old["mic_sha256"] and opened == old["external"]["delivered"]
                    if mode == "fixed" and profile in ("aac44", "sbc48") and clocks:
                        i = len(rows)
                        meta = dict(voice=s["voice"], room=s["room"], carrier=s["carrier"], cue_at=start, command_at=at)
                        ref = dict(fixed=scored["fixed"], c161=scored["c161"], events=dict(nyx=clocks))
                        row["p4"] = uplink.p4(meta, mic, detector, i, decoder, reader, ref, logs, require_timely=False)
                    rows.append(row)
            print(s["voice"], s["room"], s["carrier"], flush=True)
        for failed in sorted(work.glob("*-attempt*.gz")):
            logs.write(failed, failed.name)
    assert not pin and len(rows) == 160 and len(tracks) == 80
    archive.with_suffix(".tmp").replace(archive)
    save(OUT.with_suffix(".json"), dict(records=rows, tracks=tracks, profiles=PROFILES, alignment=delays,
         codec_libraries=av.library_versions, pyav=av.__version__, asr=identity, **firmware, archive_sha256=sha(archive),
         script_sha256=sha(Path(__file__)), source_pack_sha256=sha(handoff.COMMANDS), cue_sha256=sha(fixed.CUE),
         plan_sha256=sha(ch.PLAN), policy="Entire external file through 44.1 or 48 kHz playback. Identical stereo "
         "channels for lossy encoding, average the decoded channels, return to 16 kHz, remove only bulk delay "
         "measured on an independent noise probe. Two rooms at archived level and background. Panel speaker idle. "
         "Host sweeps fixed and silence controls. For each compressed positive with a notification, cold-boot P4 "
         "listener and encoder with zero reference, and transcribe the received Opus using unprompted GPU small.en."))


if __name__ == "__main__":
    main()
