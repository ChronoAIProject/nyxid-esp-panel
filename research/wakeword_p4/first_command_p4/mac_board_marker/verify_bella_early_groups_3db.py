"""Compare to the shipped triple cue before carrying the gain change into longer replays."""
from collections import Counter, defaultdict
import sys
from unittest.mock import patch

import check_bella_early_groups_3db as shifts
import check_bella_headroom_history as history
import check_bella_headroom_collisions as collisions
import check_third_triplet_collisions as previous

c = shifts.candidate
P = c.P
OUT = c.FIT.with_name("4090-bella-early-groups-3db-comparison.json")
HISTORY = OUT.with_stem(OUT.stem.replace("comparison","history"))
COLLISIONS = OUT.with_stem(OUT.stem.replace("comparison","collisions"))


def classify(events, start, end):
    opened = next(iter(events["O"]),None)
    return opened is not None and (start is None or start <= opened < end)


def compare():
    rows = P.read(shifts.OUT)["records"]
    triples = defaultdict(dict)
    for row in rows:
        key = row["angle"],row["direction"],row["scene"],row["offset_ms"]
        triples[key][row["role"]] = row
    summary = defaultdict(Counter)
    for (_,_,_,offset), trials in triples.items():
        assert set(trials) == {"old","new","erased"}
        a,b = (trials[role]["timely"] for role in ("old","new"))
        summary[str(offset)].update(cases=1,old=int(a),new=int(b),gained=int(b and not a),lost=int(a and not b),
                                    erased_open=int(bool(trials["erased"]["events"]["O"])))
    assert len(summary)==5 and all(count["cases"]==12 for count in summary.values())
    base = {r["label"]:r for r in P.read(previous.REPLAY)["records"]}
    report = P.read(c.REPLAY)
    assert report["build"] == P.read(previous.REPLAY)["build"]
    panels = defaultdict(Counter)
    for row in report["records"]:
        old = base.pop(row["label"])
        assert all(row[k]==old[k] for k in ("group","start","end","ref_sha256"))
        a,b = (classify(r["events"],r["start"],r["end"]) for r in (old,row))
        panels[row["group"]].update(cases=1,old=int(a),new=int(b),gained=int(b and not a),lost=int(a and not b))
    assert not base and len(report["records"]) == 59
    passed = (summary["0"]["gained"]>0 and summary["0"]["lost"]==0 and
              all(g["lost"]==0 for name,g in panels.items() if name!="synchronous") and
              panels["synchronous"]["new"]==0)
    deps = (c.PROTO,c.FIT,c.REPLAY,previous.REPLAY,shifts.OUT)
    P.save(OUT,dict(fit={"cue_sha256":P.sha(c.WAV),"original_cue_sha256":P.sha(c.OLD)}, build=report["build"],
                    shifts=summary,previous_panel=panels,history_gate=passed,sources={str(p):P.sha(p) for p in deps}))


def follow(stage):
    info = P.read(OUT)
    assert info["history_gate"] and info["fit"]["cue_sha256"]==P.sha(c.WAV)
    folder = c.WORK/c.NAME
    fit = c.triple.fit
    with patch.multiple(fit, WORK=folder, WAV=c.WAV, OUT=OUT):
        if stage == "history":
            with patch.object(history,"OUT",HISTORY):
                history.main()
        else:
            with patch.object(fit.early,"WAV",c.OLD), patch.multiple(collisions,OUT=COLLISIONS,PROTO=c.PROTO):
                collisions.main()


if __name__ == "__main__":
    action, = sys.argv[1:]
    if action == "compare":
        compare()
    else:
        assert action in ("history","collisions")
        follow(action)
