"""Lower only the panel's demonstrator by 6 dB; keep the external wake cue and room gains."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_panel_middle_p4 as middle

ch, tx = middle.ch, middle.tx
OUT = ch.RESULTS / "4090-c161-quieter-panel-demo.json"
WAVE = ch.RESULTS / "hye-nyx-panel-bca-minus6.wav"


def main():
    v5, demo = tx.study.panel_wave()
    quieter = np.rint(demo.astype(float) * 10**(-6/20)).astype("<i2")
    sf.write(WAVE, quieter, 16000, subtype="PCM_16")
    restored, rate = sf.read(WAVE, dtype="int16")
    assert rate == 16000 and np.array_equal(restored, quieter)
    matrix = ch.read(tx.diag.BOARD)
    history = ch.read(middle.OUT)
    host_base = ch.read(middle.single.OUT)
    base = {r["host_index"]: r for r in history["records"] if r["group"].startswith("balanced.")}
    builds = dict(bca=middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == history["firmware"]
    versions = matrix["versions"]
    assert tx.panel.grid.gap.uplink.firmware() == versions["codec"]
    small, ident = ch.load_reader()
    assert ident == versions["small"]
    turbo = tx.diag.reader(versions["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    rows, guest, rooms = [], [], []
    summary = defaultdict(Counter)
    scratch = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(scratch, "w", ZIP_DEFLATED) as z:
        work = Path(folder)
        matcher = tx.single.timed.binding(work)
        native = tx.panel.grid.room.rooms.c161_binding()
        ref_matcher = tx.study.reference_binding(work / "ref")
        dry_event, = tx.study.host.detect(ref_matcher, [quieter])
        assert dry_event["t"] == tx.event_time()

        def score(pcm, clock):
            raw = tx.panel.grid.fit.fixed.ch.routes(matcher, native, pcm)
            return middle.pair.admit(raw, clock, middle.single.hits(pcm, clock), np.zeros_like(pcm))

        for src in matrix["records"]:
            if (src["name"], src["role"]) != ("bca", "both"):
                continue
            i, s = src["host_index"], src["source"]
            old = base[i]
            _, noise, _, cmd, h, gain, _ = tx.diag.scene(src)
            (hp, _, _), _ = tx.diag.room(s["geometry"])
            command, _ = tx.balanced.match_level(v5, cmd)
            local = fftconvolve(quieter.astype(float), hp)
            start = s["local_at"]
            mixed = noise.copy()
            mixed[start:start+len(local)] += local
            at = s["start"]
            clock = start + dry_event["t"]
            assert clock == old["transmit_event"]
            for role, cue in (("both", v5), ("command_only", np.zeros_like(v5)), ("panel", None)):
                wave = mixed.copy()
                if cue is not None:
                    wet = fftconvolve(np.r_[cue, command].astype(float), h) * gain
                    wave[at:at+len(wet)] += wet
                x = tx.study.grid.room.quantize(wave * s["gain"])
                result = score(x, clock)
                opened = result["opened"]
                timely = bool(opened and at <= opened[0] < s["command_at"])
                rows.append(dict(host_index=i, source=s, role=role, clock=clock, baseline_index=old["index"],
                                 timely=timely, result=result))
                summary[f'host.{role}.{s["external_db"]}'].update(trials=1, opens=int(bool(opened)), timely=int(timely))
                # Include every weakest-gain command, and all source-removal controls on the old misses.
                if (role == "both" and s["external_db"] == -6) or (role != "both" and not old["opened"]):
                    case = (i, s, "bca", role, x, np.zeros_like(x), result)
                    row = tx.panel.capture(work, z, len(guest), case, small, turbo, decoder, builds, tx_pair=clock)
                    row.update(role=role, baseline_index=old["index"], transmit_event=clock)
                    guest.append(row)
                    summary["guest." + role].update(trials=1, opens=int(bool(opened)), timely=int(timely),
                                                    small=int(row["small"]), turbo=int(row["turbo"]))
        print("dual source", dict(summary), flush=True)
        waves = dict(v5=v5, bca=demo, quieter=quieter)
        for i, ((s, signals, _), old) in enumerate(zip(tx.study.host.study.established(waves),
                                                      host_base["rooms"], strict=True)):
            assert ch.digest(signals["bca"]) == old["pcm_sha256"]
            ref = tx.study.host.study.prior.level.guard.refs(s, "quieter", waves)
            events = tx.study.host.detect(ref_matcher, [ref])
            clock = events[0]["t"] if events else 0
            assert len(events) <= 1
            r = score(signals["quieter"], clock)
            rooms.append(dict(index=i, source=s, digital_cue_clock=clock, result=r))
            summary["rooms." + s["kind"]].update(trials=1, opens=int(bool(r["opened"])),
                                                 missing_playout_marker=int(not events))
            if (i+1)%112 == 0:
                print("rooms", i+1, flush=True)
    assert (len(rows), len(rooms), len(guest)) == (180, 672, 26)
    with ZipFile(scratch) as z:
        assert z.testzip() is None
    archive = OUT.with_suffix(".zip")
    scratch.replace(archive)
    ch.save(OUT, dict(records=rows, rooms=rooms, guest=guest, summary=summary, model=history["model"],
        firmware=history["firmware"], codec=versions["codec"], small=ident, turbo=versions["turbo"],
        archive_sha256=ch.sha(archive), sources={p.name: ch.sha(p) for p in
            (WAVE, tx.study.WAV, tx.diag.BOARD, middle.OUT, middle.single.OUT, Path(__file__))},
        policy="Only the panel demonstration PCM is attenuated by 6 dB before propagation. Preserve the original "
               "external v5, balanced command, hardware gains, scene clocks, music, zero electrical R and receiver. "
               "Replay all 60 geometry/gain/offset combinations as dual source, panel plus external command with "
               "no external address, and panel without external speaker. P4 covers all 20 weakest-gain commands "
               "and the two controls on each old miss. Host also covers all 672 existing self-play room conditions."))


if __name__ == "__main__":
    main()
