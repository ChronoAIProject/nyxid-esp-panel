"""Factor the known synthetic wake and background sources at the original listener OPEN."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_marker_music_commands as music

ch, grid, prior = music.ch, music.grid, music.prior
uplink = grid.gap.uplink
OUT = ch.RESULTS / "4090-c161-marker-command-stems.json"
GROUPS = ("air_conditioner", "music", "assigned_music")
PAIRS = (("full", "without_cue"), ("without_background", "command_only"))


def on_clock(wave, start, length):
    signal = np.zeros(length)
    assert 0 <= start and start + len(wave) <= length
    signal[start:start + len(wave)] = wave
    return signal


def mixes(command, background, wake, gain):
    # Preserve the order of additions and quantize once, at the microphone.
    return {name: grid.room.quantize(value * gain) for name, value in (
        ("full", (background + command) + wake),
        ("without_cue", background + command),
        ("without_background", command + wake),
        ("command_only", command),
    )}


def inputs(cue):
    common = ch.read(music.normal.OUT)["records"]
    for i, (source, _, kind, h, bg, command) in enumerate(prior.audio.room.room_inputs()):
        index, row = next((j, r) for j, r in enumerate(common) if r["scene"] == i and r["cue"] == "rms")
        wake = fftconvolve(cue.astype(float), h)
        start = source["end"] - len(cue)
        signals = mixes(command, bg, on_clock(wake, start, len(bg)), source["gain"])
        assert ch.digest(signals["without_cue"]) == source["pcm_sha256"]
        yield kind, i, music.normal.OUT, index, row, signals

    broad = [r for r in ch.read(music.study.OUT)["records"] if r["group"] == "other_music"]
    songs = {r["source_path"]: r for r in ch.read(music.study.fit.DEV)["records"]}
    bank = grid.room.rooms.source_bank()
    assigned = [r for r in ch.read(music.OUT)["records"] if r["name"] == "rms" and "assigned" in r["groups"]]
    for row in assigned:
        idx = row["board"]["source"]["grid_index"]
        record = broad[idx]
        s = record["source"]
        room, remainder = divmod(idx, 22)
        song, level = divmod(remainder, 2)
        assert room == (3 * song + 1) % 28 and level == song % 2
        h, _, room_hash = grid.room.rooms.read_room(f'h{s["room_id"]:03}_')
        song_path = Path(songs[s["track"]]["path"])
        assert room_hash == s["rir_sha256"] and ch.sha(song_path) == s["noise_sha256"]
        with sf.SoundFile(song_path) as audio:
            audio.seek(s["offset"])
            bg = audio.read(s["samples"]) * s["noise_gain"]
        command = bank[s["voice"], s["command"]]
        assert ch.digest(command) == s["command_pcm_sha256"]
        command = on_clock(fftconvolve(command.astype(float), h), s["command_at"], len(bg))
        warp = grid.drift.project(cue, Fraction(1000000, 1000000 + s["player_ppm"]))
        wake = fftconvolve(warp.astype(float), h) * 10**(s["level_db"] / 20)
        signals = mixes(command, bg, on_clock(wake, s["start"], len(bg)), s["mic_gain"])
        assert ch.digest(signals["without_cue"]) == record["control_sha256"]
        yield "assigned_music", idx, music.OUT, row["index"], row, signals


def replay_pair(work, archive, i, names, fifo, decoder):
    count = len(fifo[names[0]])
    assert count == len(fifo[names[1]])
    payload = struct.pack("<3I", 0x3552504f, i, count) + b"".join(fifo[n].tobytes() for n in names)
    log = work / f"pair-{i:03}.log"
    for attempt in range(3):
        uplink.boot(uplink.CODEC, log, b"UPLINK PASS", payload, work)
        uart = log.read_text()
        try:
            encoded = {n: uplink.packets(uart, i, count, lane) for lane, n in enumerate(names)}
            break
        except (AssertionError, ValueError):
            archive.write(log, f"retries/{log.stem}-{attempt}.log")
            if attempt == 2:
                raise
    archive.write(log, log.name)
    return {n: (packets, uplink.decode(decoder, packets)) for n, packets in encoded.items()}


def score_case(work, archive, originals, serial, data, readers, decoder):
    kind, scene, path, idx, r, signals = data
    board = r["board"]
    source = board["source"]
    assert ch.digest(signals["full"]) == board["mic_sha256"]
    log = originals[path].read(f"events/{idx:03}.txt").decode().splitlines()
    assert log == board["uart_events"]
    opened = [int(line.split()[1]) for line in log if line.startswith("OPEN ")]
    assert opened == board["opened"]
    rec = dict(group=kind, scene=scene, source=source, original_report=path.name, original_row=idx,
               opened=opened, mic_sha256={n: ch.digest(x) for n, x in signals.items()}, views={})
    if not opened:
        assert not r.get("uplink")
        return rec
    emitted, first, count, blocks, *_ = map(int, next(l for l in log if l.startswith("FINISH ")).split()[1:])
    cap = r["uplink"]
    assert count == emitted - first == cap["samples"] == cap["delayed_raw_samples"]
    assert first == max(0, opened[0] // 160 * 160 - 32000) and cap["processed_samples"] == 0
    fifo = {n: np.pad(x, (1440, blocks * 256 - len(x)))[first:emitted] for n, x in signals.items()}
    assert ch.digest(fifo["full"]) == cap["fifo_sha256"]
    old_uart = originals[path].read(f"{idx:02}-opus.log").decode().replace("\r\n", "\n")
    old_packets = uplink.packets(old_uart, idx, count, 0)
    assert len(old_packets) == cap["packets"]
    rec.update(first=first, emitted=emitted, samples=count)
    for pair_index, names in enumerate(PAIRS):
        boot_id = 2 * serial + pair_index
        waves = replay_pair(work, archive, boot_id, names, fifo, decoder)
        for lane, (name, (packets, wave)) in enumerate(waves.items()):
            scores = {k: uplink.recognition(reader, wave, source["carrier"]) for k, reader in readers.items()}
            if name == "full":
                assert packets == old_packets and ch.digest(wave) == cap["pcm_sha256"]
                turbo = cap["turbo"] if "turbo" in cap else r["views"]["turbo"]["original"]
                for k, old in (("small", cap["recognition"]), ("turbo", turbo)):
                    assert scores[k]["text"] == old["text"] and scores[k]["complete_suffix"] == old["complete_suffix"]
            rec["views"][name] = dict(boot_id=boot_id, lane=lane, fifo_sha256=ch.digest(fifo[name]),
                                      decoded_sha256=ch.digest(wave), packets=len(packets), scores=scores)
    return rec


def summarize(rows):
    totals = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        views = row["views"]
        for names in PAIRS:
            for name in names:
                stats = totals[row["group"]][name]
                stats.update(trials=1, opened=bool(row["opened"]))
                for size in ("small", "turbo"):
                    actual = bool(views and views[name]["scores"][size]["complete_suffix"])
                    old = bool(views and views["full"]["scores"][size]["complete_suffix"])
                    stats.update({size: actual, size + "_gained": actual and not old, size + "_lost": old and not actual})
    return totals


def main():
    small, identity = ch.load_reader()
    info = ch.read(music.OUT)["turbo"]
    model_dir = prior.audio.prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n, h in info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    assert small.model.device == turbo.model.device == "cuda"
    cue = music.normal.test.tracks()["rms"]
    decoder = uplink.decoder_api()
    firmware = uplink.firmware()
    rows, sources = [], {}
    for path in (music.normal.OUT, music.OUT):
        doc = ch.read(path)
        archive = path.with_suffix(".zip")
        assert doc["small"] == identity and doc["turbo"] == info and ch.sha(archive) == doc["archive_sha256"]
        sources.update({archive.name: doc["archive_sha256"], path.name: ch.sha(path)})
    archive = OUT.with_suffix(".zip")
    tmp = archive.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as logs, \
            ZipFile(music.normal.OUT.with_suffix(".zip")) as common, ZipFile(music.OUT.with_suffix(".zip")) as assigned:
        work = Path(folder)
        uplink.WORK = work
        originals = {music.normal.OUT: common, music.OUT: assigned}
        for i, data in enumerate(inputs(cue)):
            row = score_case(work, logs, originals, i, data, {"small": small, "turbo": turbo}, decoder)
            rows.append(row)
            print(i, row["group"], row["scene"], row["opened"], flush=True)
        for failure in sorted(work.glob("*.gz")):
            logs.write(failure, f"retries/{failure.name}")
    assert Counter(r["group"] for r in rows) == dict(zip(GROUPS, (8, 8, 11)))
    assert sum(bool(r["opened"]) for r in rows) == 26
    assert len({(r["group"], r["scene"]) for r in rows}) == 27
    with ZipFile(tmp) as logs:
        assert logs.testzip() is None
    tmp.replace(archive)
    for p in (music.study.OUT, music.study.fit.DEV, music.normal.test.levels.FIT, Path(__file__)):
        sources[p.name] = ch.sha(p)
    ch.save(OUT, dict(records=rows, summary=summarize(rows), archive_sha256=ch.sha(archive), sources=sources,
                      small=identity, turbo=info, cue_pcm_sha256=ch.digest(cue), codec=firmware["opus_sha256"],
                      qemu=firmware["qemu_sha256"], policy="Consumed synthetic development scenes. Condition on the "
                      "equal-RMS cue's observed OPEN, keeping the original missing trial in each denominator. Replay "
                      "the entire selected FIFO with zero playback reference. Only the known synthetic source stems "
                      "are muted before microphone quantization. Reproduce every baseline selected-lane Opus packet "
                      "and unprompted transcript. These counterfactuals measure joint codec and reader behavior "
                      "when a source is present. They require oracle stems and do not predict new wake clocks."))


if __name__ == "__main__":
    main()
