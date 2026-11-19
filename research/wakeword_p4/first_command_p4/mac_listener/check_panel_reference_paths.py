"""Replay recorded host microphone hits with filtered, delayed or missing playback references."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_panel_marker_order as study
import lcd_x8_reference as lcd

ch = study.ch
BASE = study.OUT.with_name(study.OUT.stem + "-timed.json")
OUT = BASE.with_name(BASE.stem + "-reference.json")


def paths(signal, noise):
    yield "ideal", signal
    for setting in ("quiet_line", "weak", "telephone"):
        x = lcd.channel(signal, setting, noise)
        yield setting, x
        if setting == "weak":
            for shift in (-256,256):
                y = np.zeros_like(x)
                if shift < 0:
                    y[:shift] = x[-shift:]
                else:
                    y[shift:] = x[:-shift]
                yield f"weak_shift_{shift}", y
    yield "missing", np.zeros_like(signal)


def main():
    old = ch.read(BASE)["records"]
    rows, totals = [], defaultdict(Counter)
    with TemporaryDirectory() as folder:
        work = Path(folder)
        det = (ch.binding(work/"strict.so",coded="v5"),None,ch.binding(work/"prefix.so",coded="aligned"),
               study.reference_binding(work/"panel"),None)
        for i,((meta,name,role,mic,ref),r) in enumerate(zip(study.simultaneous(),old,strict=True)):
            assert (meta,name,role) == (r["source"],r["name"],r["role"])
            base = r["result"]
            assert (ch.digest(mic),ch.digest(ref)) == (base["pcm_sha256"],base["reference_sha256"])
            cache = {base["pcm_sha256"]:(base["fixed"],base["c161"])}
            # Six arm/role combinations share one noise series per scenario.
            noise = np.random.default_rng(18221+i//6).standard_normal(len(ref))
            for setting,x in paths(ref,noise):
                result = study.score(mic,x,det,cache)
                if setting == "ideal":
                    assert base == result
                clocks = result["original_opened"] if name=="abc" else result["opened"]
                rows.append(dict(host_index=i,path=setting,reference_sha256=result["reference_sha256"],
                                 panel_pairs=result["panel_pairs"],opened=clocks))
                totals[f"{name}.{role}.{setting}"].update(trials=1,opens=bool(clocks),
                    timely=bool(role!="panel" and clocks and meta["start"]<=clocks[0]<meta["command_at"]),
                    before_external=bool(role!="panel" and clocks and clocks[0]<meta["start"]))
    assert len(rows)==360*7
    ch.save(OUT, dict(records=rows,summary=totals, sources={p.name:ch.sha(p) for p in (BASE,Path(lcd.__file__),Path(study.__file__),Path(__file__))},
            policy="Hold the acoustic scene, matched microphone events, panel order and frozen C161 constant. "
                   "Alter only the acquired reference using historical LCD-X8 line models and a one-capture-block "
                   "timing skew; include a fully missing reference as a fault control. Host clocks on reused inputs. "
                   "Neither the electrical transfer function nor the delay has been measured on the target board."))


if __name__ == "__main__":
    main()
