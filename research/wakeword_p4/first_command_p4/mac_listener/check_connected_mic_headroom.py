"""Measure the paired 24 kb/s first command at a single simulated microphone gain."""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_balanced_connected_commands as balanced

ch, mixed, tx = balanced.ch, balanced.mixed, balanced.tx
OUT = ch.RESULTS / "4090-c161-connected-mic-headroom.json"
HEADROOM_DB = 3.0


def inputs(previous):
    groups = mixed.study.sources()
    v5, _ = tx.study.panel_wave()
    quiet, rate = sf.read(mixed.panel.demo.WAVE, dtype="int16")
    assert rate == 16000 and ch.sha(mixed.panel.demo.WAVE) == previous["sources"][mixed.panel.demo.WAVE.name]
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    assert ch.sha(balanced.PLAY) == previous["speakers_sha256"]
    with ZipFile(balanced.PLAY) as pack:
        manifest = json.loads(pack.read("manifest.json"))
        tracks = {r["index"]: r for r in manifest["records"]}
        pairs = set()
        for old in previous["records"]:
            s = old["source"]
            pair = s["index"], s["external_db"]
            assert pair not in pairs
            pairs.add(pair)
            entry = tracks[s["index"]]
            track = entry["tracks"]["balanced_louder"]
            wave, rate = sf.read(BytesIO(pack.read(track["file"])), dtype="int16")
            assert rate == 16000 and ch.digest(wave) == track["pcm_sha256"]
            assert entry["command_at"] == s["command_at"]
            room = mixed.room_mix(s, groups[s["voice"], s["command"]], quiet, v5, bed, bed_sha)
            signal = room(wave)
            mic, clipped = mixed.digitize(signal, s["preamp_gain"])
            assert ch.digest(mic) == old["mic_sha256"] and clipped == old["clipped_samples"]
            wave[:s["command_at"]] = 0
            control = room(wave)
            old_control, _ = mixed.digitize(control, s["preamp_gain"])
            assert ch.digest(old_control) == old["no_address"]["pcm_sha256"]
            yield old, signal, control
        assert len(tracks) == 12 and pairs == {(i, db) for i in tracks for db in (-6, 0, 6)}


def peak_dbfs(x):
    return 20 * math.log10(float(np.max(np.abs(x))) / 32768)


def main():
    previous = ch.read(balanced.OUT)
    scenes = list(inputs(previous))
    # Size the gain once on the positive floating-point mixes, before running any recognizer.
    highest = max(peak_dbfs(signal) for _, signal, _ in scenes)
    gain_db = min(0., -HEADROOM_DB - highest)
    gain = 10**(gain_db/20)
    print(f"Restored {len(scenes)} pairs. Peak at unity {highest:.3f} dBFS, fixed gain {gain_db:.3f} dB.", flush=True)
    builds = dict(bca=mixed.panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == previous["firmware"]
    uplink = tx.panel.grid.gap.uplink
    assert uplink.firmware() == previous["codec"]
    decoder = uplink.decoder_api()
    small, identity = ch.load_reader()
    assert identity == previous["small"]
    turbo = tx.diag.reader(previous["turbo"])
    proc, model, qid = balanced.original.asr.load()
    assert qid == previous["qwen"]

    sources = {p.name: ch.sha(p) for p in (balanced.OUT, balanced.PLAY, mixed.panel.demo.WAVE,
               Path(mixed.__file__), Path(balanced.__file__), Path(tx.panel.__file__), Path(__file__))}
    revision = sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    checkpoints = Path(".local/wakeword/c161-connected-mic-headroom") / revision
    checkpoints.mkdir(exist_ok=True, parents=True)
    rows, stats = [], defaultdict(Counter)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=checkpoints) as folder:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        tmp = work / archive.name
        with ZipFile(tmp, "w", ZIP_DEFLATED) as logs:
            for old, signal, control in scenes:
                s = dict(old["source"], preamp_gain=gain)
                clock = s["transmit_event"]
                tally = stats[str(s["external_db"])]
                for role, floating in (("both", signal), ("no_address", control)):
                    mic, clipped = mixed.digitize(floating, gain)
                    zero = np.zeros_like(mic)
                    raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, mic)
                    echo = mixed.panel.middle.single.hits(mic, clock)
                    host = mixed.panel.middle.pair.admit(raw, clock, echo, zero)
                    index = len(rows)
                    case = (old["index"], s, "bca", role, mic, zero, host)
                    shard = checkpoints / f"{index:02}.zip"
                    if not shard.exists():
                        partial = work / shard.name
                        try:
                            with ZipFile(partial, "w", ZIP_DEFLATED) as case_log:
                                row = tx.panel.capture(work, case_log, index, case, small, turbo, decoder,
                                                       builds, tx_pair=clock)
                                reading = None
                                if row["opened"]:
                                    cap = row["uplink"]
                                    log = case_log.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
                                    packets = uplink.packets(log, index, cap["samples"], 0)
                                    audio = uplink.decode(decoder, packets)
                                    reading = balanced.evaluate(proc, model, audio, s["command"])
                                    assert len(packets) == cap["packets"] and reading["pcm_sha256"] == cap["pcm_sha256"]
                                    assert ch.digest(balanced.fifo_input(mic, row["finish"])) == cap["fifo_sha256"]
                                row["qwen"] = reading
                                case_log.writestr("case.json", json.dumps(row))
                        except Exception:
                            partial.replace(checkpoints / f"{index:02}-failed.zip")
                            raise
                        partial.replace(shard)
                    with ZipFile(shard) as case_log:
                        row = json.loads(case_log.read("case.json"))
                        assert (row["mic_sha256"], row["opened"], row["source"], row["role"], row["index"]) == (
                            ch.digest(mic), host["opened"], s, role, index)
                        for name in case_log.namelist():
                            if name != "case.json":
                                logs.writestr(name, case_log.read(name))
                    reading = row["qwen"]
                    row.update(peak_dbfs=peak_dbfs(floating * gain), clipped_samples=clipped,
                               previous_gain=old["source"]["preamp_gain"],
                               previous_peak_dbfs=peak_dbfs(floating * old["source"]["preamp_gain"]))
                    if role == "both":
                        timely = mixed.study.tally(tally, s, row)
                        row.update(timely=timely, previous_clipped=old["clipped_samples"], previous_opened=old["opened"])
                        tally.update(previous_clipped=old["clipped_samples"], clipped=clipped,
                                     same_open_clock=int(old["opened"] == row["opened"]))
                        for name, before, after in (("qwen", old["qwen"]["complete"], bool(reading and reading["complete"])),
                                                    ("turbo", old["turbo"], row["turbo"]),
                                                    ("small", old["small"], row["small"])):
                            tally.update({name+"_before": int(before), name: int(after), name+"_timely": int(timely and after),
                                          name+"_gained": int(after and not before), name+"_lost": int(before and not after)})
                    else:
                        tally.update(controls=1, control_opens=int(bool(row["opened"])), control_clipped=clipped)
                    rows.append(row)
                if len(rows) % 12 == 0:
                    print("P4 playbacks", len(rows), flush=True)
        assert len(rows) == 72 and all(c["trials"] == c["controls"] == 12 for c in stats.values())
        with ZipFile(tmp) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
        report = dict(records=rows, summary=stats, unity_peak_dbfs=highest, target_headroom_db=HEADROOM_DB,
                      fixed_preamp_gain=gain, fixed_preamp_db=gain_db, model=previous["model"],
                      firmware=previous["firmware"], codec=previous["codec"], qwen=qid, small=identity,
                      turbo=previous["turbo"], archive_sha256=ch.sha(tmp),
                      sources=sources,
                      policy="Reuse the 36 balanced connected development scenes and remove only the external address "
                             "for each paired control. Fit one gain to the maximum positive mix peak at -3 dBFS "
                             "before quantization and before recognition. Replay all 72 in emulated P4 with zero "
                             "reference. Score the complete 24 kb/s decoded uplink, counting closed trials as misses.")
        ch.save(work / OUT.name, report)
        tmp.replace(archive)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
