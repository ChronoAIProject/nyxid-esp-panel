"""Replay complete connected utterances and their fixed-cue pairs in the same two-source scene."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import complete_quieter_panel_matrix as panel
import connected_nick_willow as connected

tx, ch = panel.tx, panel.ch
OUT = ch.RESULTS / "4090-c161-connected-panel-rooms.json"
GUEST = OUT.with_name(OUT.stem + "-p4.json")
LEVELS = (-6, 0, 6)


def sources():
    plan = ch.read(connected.PLAN)
    assert ch.sha(connected.PACK) == plan["audio_sha256"]
    groups = defaultdict(dict)
    with ZipFile(connected.PACK) as z:
        for row in plan["records"]:
            data = z.read(row["member"])
            pcm, sr = sf.read(BytesIO(data), dtype="int16")
            assert sr == 16000 and pcm.shape == (row["samples"],)
            assert sha256(data).hexdigest() == row["sha256"] and ch.digest(pcm) == row["pcm_sha256"]
            assert ch.digest(pcm[row["end"]:]) == row["suffix_sha256"]
            groups[row["voice"], row["carrier"]][row["target_id"]] = row, pcm
    assert len(groups) == 12 and all(len(group) == 5 for group in groups.values())
    return groups


def scenes():
    v5, _ = tx.study.panel_wave()
    demo, sr = sf.read(panel.demo.WAVE, dtype="int16")
    assert sr == 16000 and ch.sha(panel.demo.WAVE) == ch.read(panel.demo.OUT)["sources"][panel.demo.WAVE.name]
    bed, _, bed_sha = tx.study.grid.room.rooms.read_noise("music")
    event = tx.event_time()
    for (voice, command), group in sources().items():
        vi, ci = connected.VOICES.index(voice), connected.CARRIERS.index(command)
        geometry = (vi + ci) % len(tx.study.GEOMETRIES)
        (hp, panel_room, ph), (hx, external_room, eh) = tx.diag.room(geometry)
        nyx, plain = group["Hey Nyx"]
        matched, coded = group["fixed-v5"]
        assert nyx["end"] == matched["end"]
        np.testing.assert_array_equal(coded[matched["start"]:matched["end"]], v5)
        np.testing.assert_array_equal(coded[matched["end"]:], plain[nyx["end"]:])
        # Keep the original file clocks. Match the panel to the literal Nyx onset.
        lag = (-3200, 0, 3200)[(vi + ci) % 3]
        panel_at = nyx["start"] - lag
        local = fftconvolve(demo.astype(float), hp)
        length = max(max(len(x) for _, x in group.values()) + len(hx) - 1, panel_at + len(local))
        length += 4800
        offset = (39413 * (3 * vi + ci)) % len(bed)
        bg = bed[(offset + np.arange(length)) % len(bed)]
        ref = fftconvolve(v5.astype(float), hp)[:len(v5)]
        bg = bg * (0.35 * np.linalg.norm(ref) / np.sqrt(len(ref)) / np.sqrt(np.mean(bg**2)))
        base = bg.copy()
        base[panel_at:panel_at+len(local)] += local
        wet = {}
        for role, (row, x) in group.items():
            cmd = x.copy()
            cmd[:row["end"]] = 0
            for key, wave in (("address", x), ("command", cmd)):
                conv = fftconvolve(wave.astype(float), hx)
                wet[role, key] = np.pad(conv, (0, length-len(conv)))
        for db in LEVELS:
            scale = 10**(db/20)
            views = {}
            for role in group:
                ext = wet[role, "address"] * scale
                views[role, "external"] = bg + ext
                views[role, "both"] = base + ext
                views[role, "no_address"] = base + wet[role, "command"] * scale
            gain = min(1., 28000/max(np.abs(x).max() for x in views.values()))
            for (role, view), signal in views.items():
                row = group[role][0]
                source = dict(index=row["index"], voice=voice, command=command, target=role,
                              start=row["start"], command_at=row["end"], view=view, external_db=db,
                              panel_at=panel_at, transmit_event=0 if view == "external" else panel_at + event,
                              geometry=geometry, panel_room=panel_room, external_room=external_room,
                              panel_rir_sha256=ph, external_rir_sha256=eh, music_sha256=bed_sha,
                              music_offset=offset, preamp_gain=gain, suffix_sha256=row["suffix_sha256"])
                yield source, tx.study.grid.room.quantize(signal * gain)


def tally(count, s, result):
    ts = result["opened"]
    timely = bool(ts and s["start"] <= ts[0] < s["command_at"])
    count.update(trials=1, opened=int(bool(ts)), timely=int(timely),
                 early=int(bool(ts and ts[0] < s["start"])), late=int(bool(ts and ts[0] >= s["command_at"])))
    return timely


def host():
    rows, summary = [], defaultdict(Counter)
    with TemporaryDirectory() as tmp:
        matcher = tx.single.timed.binding(Path(tmp))
        native = tx.panel.grid.room.rooms.c161_binding()
        for s, pcm in scenes():
            clock = s["transmit_event"]
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, pcm)
            hits = panel.middle.single.hits(pcm, clock)
            result = panel.middle.pair.admit(raw, clock, hits, np.zeros_like(pcm))
            timely = tally(summary[s["target"] + "." + s["view"]], s, result)
            rows.append(dict(source=s, result=result, mic_c=hits, timely=timely))
    assert len(rows) == 540
    current = ch.read(panel.OUT)
    ch.save(OUT, dict(records=rows, summary=summary, model=current["model"], sources={p.name: ch.sha(p)
        for p in (connected.PLAN, connected.PACK, panel.demo.WAVE, panel.OUT, Path(__file__))},
        policy="Twelve voice-command pairs, each rendered with all five utterances in one assigned room pair "
               "and one panel offset, at three external gains. All views within a pair and gain share music, "
               "time axis and preamp gain. Source PCM retains original synthesis levels and pauses. Time "
               "boundaries use the existing Kokoro token durations. Removed-address controls mute all "
               "source samples before the command, prior to room convolution. Reused development only."))


def guest():
    baseline, history = ch.read(OUT), ch.read(panel.OUT)
    builds = dict(bca=panel.middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == history["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == history["codec"]
    small, identity = ch.load_reader()
    assert identity == history["small"]
    turbo = tx.diag.reader(history["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    rows, summary = [], defaultdict(Counter)
    tmp = GUEST.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp, "w", ZIP_DEFLATED) as z:
        for i, ((s, pcm), old) in enumerate(zip(scenes(), baseline["records"], strict=True)):
            assert s == old["source"] and ch.digest(pcm) == old["result"]["pcm_sha256"]
            # One gain per family, balanced across voices and commands, chosen without model outputs.
            assigned = LEVELS[(connected.VOICES.index(s["voice"]) + connected.CARRIERS.index(s["command"])) % 3]
            if s["external_db"] != assigned or (s["view"] != "both" and s["target"] != "Hey Nyx"):
                continue
            case = (i, s, "bca", s["view"], pcm, np.zeros_like(pcm), old["result"])
            row = tx.panel.capture(Path(folder), z, len(rows), case, small, turbo, decoder, builds,
                                   tx_pair=s["transmit_event"])
            count = summary[s["target"] + "." + s["view"]]
            timely = tally(count, s, row)
            count.update(small=int(row["small"]), turbo=int(row["turbo"]), timely_turbo=int(timely and row["turbo"]))
            rows.append(dict(row, timely=timely))
            if len(rows) % 7 == 0:
                print("P4", len(rows), flush=True)
    assert len(rows) == 84
    with ZipFile(tmp) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
        assert sum(n.startswith("events/") for n in z.namelist()) == len(rows)
    archive = GUEST.with_suffix(".zip")
    tmp.replace(archive)
    ch.save(GUEST, dict(records=rows, summary=summary, host_sha256=ch.sha(OUT), archive_sha256=ch.sha(archive),
        firmware=history["firmware"], model=history["model"], codec=history["codec"], small=identity, turbo=history["turbo"],
        policy="All five utterances with the panel, plus literal Nyx without panel and panel with only its "
               "command. One assigned gain for each family. Verify firmware events and capture all opened Opus."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("host", "guest"))
    (host if parser.parse_args().mode == "host" else guest)()
