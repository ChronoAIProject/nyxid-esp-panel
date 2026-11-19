"""Remove one interfering source at a time on paired fixed-cue misses, at the original gain."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_panel_rms_cue as study

ch, tx = study.ch, study.tx
OUT = ch.RESULTS / "4090-c161-panel-cue-masking.json"


def timely(clocks, scene):
    return bool(clocks and scene["start"] <= clocks[0] < scene["command_at"])


def main():
    base = {r["host_index"]: r for r in ch.read(study.middle.OUT)["records"]
            if r["group"].startswith("balanced.")}
    rms = {r["host_index"]: r for r in ch.read(study.OUT)["records"] if r["cue"] == "rms"}
    assert base.keys() == rms.keys() and len(base) == 60
    sources = ch.read(tx.diag.BOARD)["records"]
    original, _ = tx.study.panel_wave()
    modified, rate = sf.read(study.WAVE, dtype="int16")
    assert rate == 16000
    records = []
    with TemporaryDirectory() as folder:
        detector = tx.single.timed.binding(Path(folder))
        native = tx.panel.grid.room.rooms.c161_binding()
        for source in sources:
            if (source["role"], source["name"]) != ("both", "bca"):
                continue
            i, scene = source["host_index"], source["source"]
            if all(timely(r[i]["opened"], scene) for r in (base, rms)):
                continue
            _, bg, local, cmd, room, gain, _ = tx.diag.scene(source)
            command, _ = tx.balanced.match_level(original, cmd)
            at, start = scene["start"], scene["local_at"]
            for tag, cue, previous in (("original", original, base[i]), ("rms", modified, rms[i])):
                external = fftconvolve(np.r_[cue, command].astype(float), room) * gain
                for music, panel in ((True, True), (False, True), (True, False), (False, False)):
                    mic = bg.copy() if music else np.zeros_like(bg)
                    if panel:
                        mic[start:start+len(local)] += local
                    mic[at:at+len(external)] += external
                    x = tx.study.grid.room.quantize(mic * scene["gain"])
                    raw = tx.panel.grid.fit.fixed.ch.routes(detector, native, x)
                    # Keep TX scheduled in every row, including the acoustically isolated panel.
                    clock = previous["transmit_event"]
                    hits = study.middle.single.hits(x, clock)
                    result = study.middle.pair.admit(raw, clock, hits, np.zeros_like(x))
                    if music and panel:
                        assert (raw["pcm_sha256"], result["opened"]) == (previous["mic_sha256"], previous["opened"])
                    records.append(dict(host_index=i, geometry=scene["geometry"], offset=scene["offset"],
                                        external_db=scene["external_db"], cue=tag, music=music, panel=panel,
                                        tx=clock, start=at, command_at=scene["command_at"], raw=raw,
                                        mic_c_hits=hits, opened=result["opened"], timely=timely(result["opened"],scene)))
    ch.save(OUT, dict(records=records, sources={p.name: ch.sha(p) for p in
        (study.middle.OUT, study.OUT, tx.diag.BOARD, study.WAVE, Path(__file__))},
        policy="Select each scene with a late or missed original or equal-RMS cue on the paired simulated P4 "
               "matrix. Native host rerun on unaltered hardware gains, room, balanced command, physical R=0 and "
               "TX clock. Cross panel acoustically audible/isolated with music audible/silent. Both-source "
               "inputs must reproduce the actual firmware microphone hashes and OPENs. Development failures only."))


if __name__ == "__main__":
    main()
