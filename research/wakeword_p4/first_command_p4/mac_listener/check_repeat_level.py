"""Separate total playback energy from duplicate marker placement on the same 672 inputs."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import rearm_repeat_reference as guard
import select_repeat_level as fit

ch, grid, repeat = guard.ch, guard.grid, guard.repeat
OUT = ch.RESULTS / (fit.STEM + "-grid.json")


def main():
    plan = ch.read(fit.PLAN)
    selected, rate = sf.read(fit.WAV, dtype="int16")
    assert rate == 16000 and ch.sha(fit.WAV) == plan["audio"]["sha256"]
    original = grid.fit.original()[1].astype(float)
    full = repeat.tracks()["repeat_trained"].astype(float)
    ratio = np.linalg.norm(full) / np.linalg.norm(original)
    boosted = np.rint(original * ratio)
    assert np.abs(boosted).max() <= 32767
    full_equal = np.rint(full / ratio).astype("<i2")
    assert ch.digest(full_equal) == plan["records"]["0"]["pcm_sha256"]
    waves = dict(selected=selected, full_equal=full_equal, boosted_v5=boosted.astype("<i2"))
    native = grid.room.rooms.c161_binding()
    original_grid = ch.read(repeat.OUT)["records"]
    stats, rows = defaultdict(Counter), []

    with TemporaryDirectory() as folder:
        detector = ch.binding(Path(folder) / "v5.so", coded="v5")
        pair = ch.binding(Path(folder) / "pair.so", coded="aligned")
        for ids in (grid.fit.EVAL, repeat.source.EXTRA):
            for source, signals, control in grid.scenes(ids, waves):
                baseline = original_grid[len(rows)]
                assert source == baseline["source"] and ch.digest(control) == baseline["control_sha256"]
                base = baseline["routes"]["v5"]
                assert ch.digest(signals["v5"]) == base["pcm_sha256"]
                results = {}
                for name in waves:
                    result = grid.outcome(detector, native, signals[name], source)
                    ref = guard.refs(source, name, waves)
                    old = repeat.drift.guarded(result, detector, pair, ref)
                    new = guard.revised(detector, result, old, ref)
                    stats[name].update(trials=1, timely=result["timely"], old_self=bool(old["opened"]),
                                       new_self=bool(new["opened"]), gained=result["timely"] and not base["timely"],
                                       lost=base["timely"] and not result["timely"],
                                       earlier=base["timely"] and result["timely"] and result["opened"] < base["opened"])
                    results[name] = dict(external=result, original_reference=old, rearmed_reference=new)
                rows.append(dict(source=source, routes=results))
            print(len(rows), flush=True)
    assert len(rows) == 672
    ch.save(OUT, dict(records=rows, summary=stats, original_to_full_rms_ratio=ratio,
                      pcm_sha256={name:ch.digest(wave) for name,wave in waves.items()},
                      sources={p.name:ch.sha(p) for p in (fit.PLAN, fit.WAV, repeat.OUT, Path(guard.__file__), Path(__file__))},
                      policy="Match room, onset, command, background, gain and rate to the published paired grid. "
                      "The boosted original matches full-repeat total RMS. Full and weak repeats at equal RMS match "
                      "the baseline. Dry pronunciation selected the weak repeat before these scores."))


if __name__ == "__main__":
    main()
