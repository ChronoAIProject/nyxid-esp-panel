"""Package the shared noisy 058 positive with original source levels and both decoded calls."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_bella_daily_sender as daily_sender
import package_weak_gain_room_audio as room
import run_guest

daily, P = room.daily, daily_sender.P
OUT = P.RESULTS / "4090-fixed-willow-058-room.json"
PACK = OUT.with_suffix(".zip")
RUNTIME = Path(".local/wakeword/c161-fixed-sound-runtime")
PROFILE = P.RESULTS / "4090-fixed-sound-runtime-replay.json"


def read_lanes(work, store, codec_zip, sender_zip, role, capture, sender):
    helper, prior = daily_sender.sender, daily_sender.prior
    label = f"058/{role}"
    opus = helper.archived(codec_zip, label + "/opus.log", capture["opus_uart_sha256"])
    packets = prior.packets(opus, 58, capture["samples"], 0)
    log = work / "sender.log"
    log.write_bytes(sender_zip.read(label + "/sender.log"))
    meta = sender["sender"]
    assert P.sha(log) == meta["uart_sha256"] and capture["fifo_sha256"] == meta["fifo_sha256"]
    lanes = helper.compare(log, 58, sender["opened"], capture["first"], capture["samples"], packets)
    decoder = prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == P.read(daily_sender.OUT)["decoder"]
    records = {}
    for condition, (sent, timing) in zip(("immediate", "waited"), lanes, strict=True):
        pcm = prior.decode(decoder, sent)
        known = meta[condition]
        assert P.digest(pcm) == known["qwen"]["pcm_sha256"]
        name = f"received/{role}-{condition}.wav"
        store(name, room.wav(pcm, "PCM_16"))
        records[condition] = dict(file=name, pcm_sha256=P.digest(pcm), first_pcm=timing["first_pcm_sample"],
                                  first_packet_ms=timing["first_send_pts"], first_captured_ms=timing["first_captured_pts"],
                                  qwen=known["qwen"]["text"], turbo=known["turbo"]["text"])
    return records


def main():
    uplink = daily_sender.daily
    sources = [Path(__file__), daily.OUT, uplink.OUT, daily_sender.OUT, uplink.official.OUT, PROFILE,
               daily.matched.PACK, daily.board.weak.PACK, room.MUSIC]
    cohorts = P.read(daily.OUT), P.read(uplink.official.OUT)
    nyx = {r["role"]:r for r in cohorts[0]["records"] if r["index"] == 58}
    willow = {r["role"]:r for r in cohorts[1]["records"] if r["index"] == 58}
    willow_positive = [r for r in cohorts[1]["records"] if r["role"] == "willow" and r["opened"] is not None]
    assert len(willow_positive) == 1 and willow_positive[0]["index"] == 58
    sent = {r["role"]:r for r in P.read(daily_sender.OUT)["records"] if r["index"] == 58}
    capture = {r["role"]:r for r in P.read(uplink.OUT)["records"] if r["index"] == 58}
    for source in (uplink.OUT, daily_sender.OUT):
        assert P.sha(source.with_suffix(".zip")) == P.read(source)["archive_sha256"]
    original = nyx["original"]
    grid, taps = daily.rooms()
    (_, scene, render, ref, active, tracks), = [c for c in grid[original["db"]] if c[0]["file"] == original["file"]]
    source, _, _, speech = next(v for v in daily.event_inputs("daily") if int(v[1]) == 58)
    assert source["text"] == original["text"] and P.digest(speech) == original["dry_speech_sha256"]
    talk, level = daily.tx.balanced.match_level(active, speech)
    assert level["gain_db"] == original["speech_gain_db"] and P.digest(ref) == original["reference_sha256"]
    with ZipFile(daily.matched.PACK) as z:
        item, = [r for r in json.loads(z.read("manifest.json"))["records"] if r["file"] == original["file"]]
        willow_wave = daily.study.playback.read_wav(z, item)
    end = original["command_at"]
    assert np.array_equal(willow_wave[end:], tracks["original"][end:])
    zeros = daily.board.bella.rear_filter(np.zeros(len(willow_wave)), taps)
    mixed_bed, _ = render(zeros)
    pure_panel, _ = render(zeros, include_music=False)
    # Split out the effective bed after ADC rounding; the solo music has up to one LSB rounding error.
    music = mixed_bed.astype(np.int32) - pure_panel.astype(np.int32)
    assert np.abs(music).max() < 32768
    guest = RUNTIME / "replay/fixed/board.bin"
    profile = P.read(PROFILE)
    assert P.sha(guest) == profile["guest_sha256"]["fixed"] and P.sha(P.QEMU) == profile["qemu_sha256"]
    sim = Path(".local/wakeword/c161-panel-middle-uart/build")
    config = P.read(sim/"flasher_args.json")
    names, observations = {}, []
    with TemporaryDirectory(dir=RUNTIME) as temp:
        work = Path(temp)
        with ZipFile(work/PACK.name,"w",ZIP_DEFLATED) as pack, ZipFile(uplink.OUT.with_suffix(".zip")) as codec, ZipFile(daily_sender.OUT.with_suffix(".zip")) as sends:
            def store(name, data):
                assert name not in names
                pack.writestr(name, data)
                names[name] = dict(bytes=len(data), sha256=sha256(data).hexdigest())
            store("music-source-LICENSE.txt", room.LICENSE.read_bytes())
            store("music-attribution.txt", (room.CREDIT+"\n").encode())
            store("play/music-at-mic-level.wav", room.wav(music.astype("<i2"), "PCM_16"))
            store("play/panel-at-source-level.wav", room.wav(ref, "PCM_16"))
            suffix = None
            for role, dry in (("original",tracks["original"]), ("willow",willow_wave), ("absent",tracks["absent"])):
                external = dry.astype(np.int32)
                external[original["speech_start"]:original["speech_end"]] += talk.astype(np.int32)
                if suffix is None:
                    suffix = external[end:]
                assert np.array_equal(suffix, external[end:]) and np.abs(external).max() < 65536
                dry_wav = room.wav(external * 32768, "PCM_24")
                back, rate = sf.read(BytesIO(dry_wav), dtype="int32")
                assert rate == 16000 and np.array_equal(back // 32768, external)
                store(f"play/{role}.wav", dry_wav)
                mic, clipped = render(daily.board.bella.rear_filter(external, taps))
                expected = willow[role]["pcm_sha256"] if role == "willow" else nyx[role]["mic_sha256"]
                assert P.digest(mic) == expected
                store(f"digital/{role}-MR.wav", room.wav(np.column_stack((mic,ref)), "PCM_16"))
                payload = run_guest.payload(mic.tobytes(), ref.tobytes())
                block = payload[:4] + struct.pack("<I",1) + payload[4:]
                image = bytearray([255])*(4<<20)
                for addr, member in config["flash_files"].items():
                    data = (guest if member == config["app"]["file"] else sim/member).read_bytes()
                    offset = int(addr,16)
                    assert offset + len(data) <= 2<<20
                    image[offset:offset+len(data)] = data
                image[2<<20:(2<<20)+len(block)] = block
                flash = work/"flash.bin"
                flash.write_bytes(image)
                log = work/(role+".log")
                run_guest.boot(P.QEMU, flash, log)
                uart = log.read_text()
                assert "Guru Meditation" not in uart and "watchdog got triggered" not in uart
                hits = [list(map(int,l.split()[2:])) for l in uart.splitlines() if l.startswith("PROFILE event ")]
                assert all(epoch == model == 0 for epoch,model,_,_ in hits)
                store(f"uart/fixed-{role}.log", log.read_bytes())
                row = dict(role=role, mic_sha256=expected, clipped=clipped, fixed_cues=[h[2] for h in hits],
                           fixed_open=hits[0][3] if hits else None)
                if role in nyx:
                    assert row["fixed_cues"] == nyx[role]["events"]["C"]
                if role in willow:
                    row["official_willow_open"] = willow[role]["opened"]
                if role in ("original", "willow"):
                    known = sent[role]
                    assert known["opened"] == (row["fixed_open"] if role == "original" else row["official_willow_open"])
                    row["received"] = read_lanes(work, store, codec, sends, role, capture[role], known)
                observations.append(row)
            def dbfs(pcm):
                return float(20*np.log10(np.sqrt(np.mean(np.asarray(pcm,dtype=float)**2))/32768))
            manifest = dict(records=observations, files=names, command=original["command"], command_at=end,
                interrupting_speech=source["text"], sample_rate=16000, source_file=original["file"], case=58,
                samples=len(ref), panel_start=scene["panel_at"], music_rms_dbfs=dbfs(music),
                panel_rms_dbfs=dbfs(pure_panel), command_mix_sha256=P.digest(suffix), reference_sha256=P.digest(ref),
                fixed_guest_sha256=P.sha(guest), qemu_sha256=P.sha(P.QEMU), sources={p.name:P.sha(p) for p in sources},
                archive_inputs={p.with_suffix(".zip").name:P.read(p)["archive_sha256"] for p in (daily_sender.OUT,uplink.OUT)},
                scope="Retrospectively selected as the only official Willow opening in 96 consumed overlapping "
                      "utterances. Fixed-nyx newly replayed on all three M/R pairs. Official Willow used classic ESP32 "
                      "QEMU. Receive WAVs decoded from archived P4 sender packets.",
                playback="Start one external speech track, music and panel tracks together. Keep playback positions "
                         "and gains fixed for all three addresses. External speech is PCM24 at half dry gain; music is "
                         "at simulated microphone level and panel is at original source level. Calibrate the isolated "
                         "music and panel levels at the mic using the full-window RMS values. Stereo digital is M then R.")
            pack.writestr("manifest.json", json.dumps(manifest, indent=2)+"\n")
        with ZipFile(work/PACK.name) as z:
            assert set(z.namelist()) == set(names) | {"manifest.json"} and z.testzip() is None
            for name, meta in names.items():
                assert sha256(z.read(name)).hexdigest() == meta["sha256"]
        P.save(work/OUT.name, dict(archive=PACK.name, archive_sha256=P.sha(work/PACK.name), **manifest))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
