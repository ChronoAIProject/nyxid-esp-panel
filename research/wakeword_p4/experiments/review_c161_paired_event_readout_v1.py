"""Verify whole-stream events and review matched targets, prefixes and hybrid routing."""
from collections import defaultdict

import numpy as np

from audit_c161_history_paths_v1 import arbitrate, decode, times
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-paired-event-readout"


def summarize():
    plan = read(RESULTS / "4090-c161-willow-matched-scenes-run-plan-v1.json")["records"]
    timing = read(RESULTS / "4090-c161-pronunciation-and-timing-review-v1.json")["matched"]
    official = read(RESULTS / "4090-c161-willow-matched-scenes-review-v1.json")["records"]
    metrics = defaultdict(lambda: defaultdict(dict))
    baseline_digests = {}
    for cohort in ("context", "daily", "willow", "mac", "joe", "generated"):
        report = read(RESULTS / f"{STEM}-{cohort}-v1.json")
        refs = read(RESULTS / f"4090-c161-vowel-name-{cohort}-v1.json")
        new = RESULTS / f"{STEM}-{cohort}-scores-v1.npz"
        old = RESULTS / f"4090-c161-vowel-name-{cohort}-scores-v1.npz"
        assert sha(new) == report["scores_sha256"] and sha(old) == refs["scores_sha256"]
        baseline_digests[cohort] = sha(old)
        with np.load(new) as traces, np.load(old) as reference:
            for r, prev in zip(report["records"], refs["records"], strict=True):
                k = r["key"]
                assert r["pcm_sha256"] == prev["pcm_sha256"]
                for suffix in ("_clock", "_c161"):
                    np.testing.assert_array_equal(reference[k + suffix], traces[k + suffix])
                clock = traces[k + "_clock"]
                decoded = {name: [decode(path, clock) for path in traces[k + "_" + name]]
                           for name in report["models"]}
                paths = {n: times(arbitrate(*ev)[0]) for n, ev in decoded.items()}
                for n in paths:
                    assert paths[n] == r["events"][n]["pair"]
                for name in ("silence", "speech"):
                    paths["c161_plus_" + name] = times(arbitrate(decoded["c161"][0], decoded[name][1])[0])
                for n, events in paths.items():
                    g = metrics[n]
                    if cohort == "context" and "timeline" in r:
                        for span in r["timeline"]:
                            if span["kind"] != "clip":
                                continue
                            kind = "connected" if span["group"] == "nyx_connected_command" else "ordinary_in_stream"
                            hits = [t for t in events if span["start"] < t <= span["end"]]
                            g[kind][f'{r["stream"]}/{span["text_index"]}'] = hits
                    elif cohort == "context":
                        group = "isolated" if r["group"] == "nyx_connected_command" else "ordinary_isolated"
                        g[group][k] = events
                    elif cohort == "willow":
                        s, a, o = plan[int(k)], timing[int(k)], official[int(k)]
                        assert s["pcm_sha256"] == r["pcm_sha256"]
                        assert (s["voice"], s["scene"], s["target_id"]) == (a["voice"], a["scene"], a["target"])
                        assert o["voice"] == a["voice"] and o["scene"] == a["scene"] and o["target"] == a["target"]
                        hits = [t for t in events if r["start"] < t <= r["end"]]
                        g["matched_" + r["role"]][k] = hits
                        prefix = [t for t in events if t <= s["target_start"]]
                        prefix_id = s["prefix_pcm_sha256"]
                        assert g["shared_prefix"].get(prefix_id, prefix) == prefix
                        g["shared_prefix"][prefix_id] = prefix
                        if r["role"] == "nyx":
                            g["command_lag_samples"][k] = [t-a["command_phone_sample"] for t in hits]
                        else:
                            g["official_willow"][k] = [t for t in o["events"]["willow"] if r["start"] < t <= r["end"]]
                    elif cohort == "daily":
                        g["daily"][k + " " + r["text"]] = events
                    elif cohort in ("joe", "mac"):
                        group = cohort + ("_target" if r["target"] == "hey_nyx" else "_other")
                        g[group][k] = events
                    elif r["role"] == "transfer":
                        g["transfer_" + r["kind"]][k] = events
    summary = {n: {k: [sum(bool(x) for x in cases.values()), len(cases)] for k, cases in groups.items()}
               for n, groups in metrics.items()}
    save(RESULTS / f"{STEM}-review-v1.json", dict(baseline_scores=baseline_digests, counts=summary, samples=metrics))
    for n, groups in metrics.items():
        hits = {key for key, ev in groups["connected"].items() if ev}
        orig = {key for key, ev in metrics["c161"]["connected"].items() if ev}
        lag = [v[0]/16 for v in groups["command_lag_samples"].values() if v]
        print(n, summary[n], "gains", sorted(hits-orig), "losses", sorted(orig-hits), "first lag ms", lag)


if __name__ == "__main__":
    summarize()
