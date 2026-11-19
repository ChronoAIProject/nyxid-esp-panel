"""Pair all changed captures and verify the published Opus baseline on identical OPENs."""
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import check_final_marker_p4 as board

ch, grid = board.ch, board.grid
OUT = ch.RESULTS/"4090-c161-final-marker-command-comparison.json"


def details(row):
    s = row["board"]["source"]
    opened = row["board"]["opened"]
    cap = row.get("uplink")
    return dict(opened=opened, timely=bool(opened and s["cue_at"] <= opened[0] < s["command_at"]),
                small=bool(cap and cap["recognition"]["complete_suffix"]), turbo=bool(cap and cap["turbo"]["complete_suffix"]))


def check_equal(left, right, baseline, trial):
    a, b = left["row"], right["row"]
    assert a["board"]["mic_sha256"] == b["board"]["mic_sha256"]
    assert a["board"]["source"]["command_at"] == b["board"]["source"]["command_at"]
    if a["board"]["opened"] != b["board"]["opened"] or not a["board"]["opened"]:
        return 0
    ac, bc = a["uplink"], b["uplink"]
    assert all(ac[k] == bc[k] for k in ("samples", "fifo_sha256", "pcm_sha256", "packets"))
    assert details(a) == details(b)
    def packets(z, idx):
        text = z.read(f"{idx:02}-opus.log").decode().replace("\r\n", "\n")
        return grid.gap.uplink.packets(text, idx, ac["samples"], 0)
    assert packets(baseline, left["index"]) == packets(trial, right["index"])
    return ac["packets"]


def main():
    scored = ch.read(board.OUT)
    candidates = scored["records"]
    rows, summary = [], defaultdict(Counter)
    external = {(r["board"]["source"]["group"],r["board"]["source"]["scene"],r["mode"]):r for r in candidates if not r["local"]}
    prior_paths = (board.prior.prior.OUT, board.prior.server.strong.OUT)
    baseline_index = {}
    for path in prior_paths:
        doc = ch.read(path)
        assert ch.sha(path.with_suffix(".zip")) == doc["archive_sha256"]
        for i, r in enumerate(doc["records"]):
            if r["name"] != "v5" or r["local"]:
                continue
            src = r["board"]["source"]
            if path == prior_paths[0] and r["group"] == "common":
                key = "common", src["scene"]
            elif path == prior_paths[1] and r["group"] == "music":
                key = "assigned", src["index"]
            else:
                continue
            baseline_index[key] = dict(index=i, path=path, row=r)
    assert len(baseline_index) == 27
    assert ch.sha(board.OUT.with_suffix(".zip")) == scored["archive_sha256"]
    with ZipFile(board.OUT.with_suffix(".zip")) as trial:
        for (group, scene, mode), row in external.items():
            if mode != "candidate":
                continue
            s = row["board"]["source"]
            cases = []
            if s["changed"] and group != "common":
                old = external[group, scene, "original"]
                cases.append((group+"_changed", dict(index=old["index"], path=board.OUT, row=old)))
            if s["assigned"] or group == "common":
                tag = "assigned" if s["assigned"] else "common"
                cases.append((tag, baseline_index[tag, scene]))
            for tag, baseline in cases:
                with ZipFile(baseline["path"].with_suffix(".zip")) as z:
                    equal = check_equal(baseline, dict(index=row["index"],row=row), z, trial)
                pair = [details(baseline["row"]), details(row)]
                c = summary[tag]
                c.update(trials=1, same_open=pair[0]["opened"]==pair[1]["opened"], identical_packets=equal)
                for name in ("timely","small","turbo"):
                    a,b = pair[0][name],pair[1][name]
                    c.update({name+"_original":a, name+"_candidate":b, name+"_gained": b and not a, name+"_lost":a and not b})
                rows.append(dict(group=tag, scene=scene, baseline_report=baseline["path"].name, baseline_index=baseline["index"],
                                 candidate_index=row["index"], original=pair[0], candidate=pair[1], equal_packets=equal))
    assert {k:c["trials"] for k,c in summary.items()} == dict(common=16, assigned=11, established_changed=24, other_music_changed=20)
    ch.save(OUT, dict(summary=summary, records=rows, sources={p.name:ch.sha(p) for p in (*prior_paths,board.OUT,Path(__file__))},
                      policy="Compare original v5 on the same physical sample clock. Reuse prior original cases in the "
                      "prespecified sets, verify received payload bytes whenever both OPEN on the same sample. Count "
                      "missed opens in the denominator. Changed-clock subsets are reported separately and can overlap "
                      "with the preassigned eleven music cases."))


if __name__ == "__main__":
    main()
