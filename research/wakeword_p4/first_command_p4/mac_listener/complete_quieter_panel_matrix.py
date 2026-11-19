"""Complete the other two gain bands on P4, preserving the 20 existing weak-gain captures."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_quieter_panel_demo as demo

ch, tx, middle = demo.ch, demo.tx, demo.middle
OUT = ch.RESULTS / "4090-c161-quieter-panel-complete-p4.json"


def main():
    first = ch.read(demo.OUT)
    host = {r["host_index"]: r for r in first["records"] if r["role"] == "both"}
    existing = {r["host_index"]: r for r in first["guest"] if r["role"] == "both"}
    assert len(host) == 60 and len(existing) == 20
    earlier = demo.OUT.with_suffix(".zip")
    assert ch.sha(earlier) == first["archive_sha256"]
    builds = dict(bca=middle.WORK / "build")
    assert ch.sha(builds["bca"] / "nyx_mac_listener.bin") == first["firmware"]
    assert tx.panel.grid.gap.uplink.firmware() == first["codec"]
    small, ident = ch.load_reader()
    assert ident == first["small"]
    turbo = tx.diag.reader(first["turbo"])
    decoder = tx.panel.grid.gap.uplink.decoder_api()
    v5, _ = tx.study.panel_wave()
    panel, rate = sf.read(demo.WAVE, dtype="int16")
    assert rate == 16000 and ch.sha(demo.WAVE) == first["sources"][demo.WAVE.name]
    summary, rows = defaultdict(Counter), []
    scratch = OUT.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(scratch, "w", ZIP_DEFLATED) as z, ZipFile(earlier) as old:
        assert old.testzip() is None
        work = Path(folder)
        for src in ch.read(tx.diag.BOARD)["records"]:
            if (src["name"], src["role"]) != ("bca", "both"):
                continue
            i, scene = src["host_index"], src["source"]
            target = host[i]
            assert target["source"] == scene
            if i in existing:
                row = existing[i]
                assert scene["external_db"] == -6
                origin = earlier.name
                assert f'events/{row["index"]:03}.txt' in old.namelist()
            else:
                _, noise, _, command, h, gain, _ = tx.diag.scene(src)
                (hp, _, _), _ = tx.diag.room(scene["geometry"])
                command, _ = tx.balanced.match_level(v5, command)
                external = fftconvolve(np.r_[v5, command].astype(float), h) * gain
                local = fftconvolve(panel.astype(float), hp)
                mic = noise.copy()
                at, local_at = scene["start"], scene["local_at"]
                mic[local_at:local_at+len(local)] += local
                mic[at:at+len(external)] += external
                x = tx.study.grid.room.quantize(mic * scene["gain"])
                expected = target["result"]
                case = (i, scene, "bca", "both", x, np.zeros_like(x), expected)
                row = tx.panel.capture(work, z, len(rows), case, small, turbo, decoder, builds, tx_pair=target["clock"])
                origin = OUT.with_suffix(".zip").name
                print(len(rows), i, row["opened"], row["small"], row["turbo"], flush=True)
            assert row["opened"] == target["result"]["opened"] and row["mic_sha256"] == target["result"]["pcm_sha256"]
            rows.append(dict(row, archive=origin, transmit_event=target["clock"]))
            summary[str(scene["external_db"])].update(trials=1, opens=int(bool(row["opened"])),
                timely=int(target["timely"]), small=int(row["small"]), turbo=int(row["turbo"]))
    assert len(rows) == len(host)
    with ZipFile(scratch) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    archive = OUT.with_suffix(".zip")
    scratch.replace(archive)
    ch.save(OUT, dict(records=rows, summary=summary, archives={earlier.name: ch.sha(earlier), archive.name: ch.sha(archive)},
        model=first["model"], firmware=first["firmware"], codec=first["codec"], small=ident, turbo=first["turbo"],
        sources={p.name: ch.sha(p) for p in (demo.OUT, demo.WAVE, tx.diag.BOARD, Path(__file__))},
        policy="All 60 two-source command cases with -6 dB panel demo. Keep 20 prior weakest-gain captures "
               "at their original archive and index. Replay all 40 remaining cases on identical firmware."))


if __name__ == "__main__":
    main()
