"""Recompute fixed five-frame events, shared arbitration and clip identities from public scores."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
MODELS = ["c161", "gain", "ambient", "pair"]
COHORTS = ["builtin", "daily", "joe", "mac", "continuous", "primed", "jenny", "melo"]


def load(name):
    return json.loads((RESULTS / name).read_bytes())


def archive(name, expected):
    path = RESULTS / name
    with path.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == expected
    return np.load(path, allow_pickle=False)


def decode(codes, clock):
    assert codes.dtype == np.uint8 and codes.shape == clock.shape
    sums = np.r_[0, np.cumsum(codes, dtype=np.int64)]
    sums = sums[5:] - sums[:-5]
    last, armed, events = -1000, True, []
    for step, value in enumerate(sums, 5):
        if step - last < 67:
            continue
        if value * 2 < 1175:
            armed = True
        if armed and value >= 1175:
            events.append(dict(step=step, relative_sample=int(clock[step-1])))
            last, armed = step, False
    return events


def arbitrate(first, second):
    candidates = [(e["step"], rank, e) for rank, group in enumerate([first, second]) for e in group]
    emitted, suppressed, last = [], [], -1000
    for step, rank, e in sorted(candidates):
        entry = dict(**e, model=["c161", "ambient"][rank])
        if step - last >= 67:
            emitted.append(entry)
            last = step
        else:
            suppressed.append(entry)
    return emitted, suppressed


def times(events):
    return [e["relative_sample"] for e in events]


def count(table, key, hits):
    counts = table[key]
    counts["clips"] += 1
    for model, hit in hits.items():
        counts[model] += int(hit)
        if model != "c161":
            counts[model+"_gained"] += int(hit and not hits["c161"])
            counts[model+"_lost"] += int(hits["c161"] and not hit)


def context():
    paired = load("4090-c161-bn-natural-language-original-v1.json")
    ambient = load("4090-c161-ambient-history-v1.json")
    gain = load("4090-c161-history-gain-development-v1.json")
    table = defaultdict(Counter)
    details = []
    with archive("4090-c161-bn-natural-language-scores-v1.npz", paired["scores_sha256"]) as base, \
         archive("4090-c161-ambient-history-scores-v1.npz", ambient["scores_sha256"]) as alt, \
         archive("4090-c161-history-gain-development-scores-v1.npz", gain["gain_scores_sha256"]) as ga:
        def events(key):
            clock = base[key+"_clock"]
            c = decode(base[key+"_c161"], clock)
            a = decode(alt[key+"_scores"], clock)
            g = decode(ga[key], clock)
            selected, _ = arbitrate(c, a)
            return {m:times(ev) for m, ev in zip(MODELS, (c,g,a,selected), strict=True)}
        streams = {stream["name"]:events(stream["name"]+"_stream") for stream in ambient["streams"]}
        for a,g in zip(ambient["records"],gain["records"],strict=True):
            name, index = a["stream"], a["index"]
            assert (name,index,a["pcm_sha256"]) == (g["stream"],g["index"],g["pcm_sha256"])
            start, end = a["start"], a["end"]
            isolated = events(f"{name}_{index}")
            continuous = {m:[t for t in ev if start<t<=end] for m,ev in streams[name].items()}
            for setting,ports in [("isolated",isolated),("continuous",continuous)]:
                for model in ("c161","ambient"):
                    assert a[setting][model] == ports[model]
                assert g[setting]["gain"] == ports["gain"]
                group = "target" if a["group"] == "nyx_connected_command" else "other"
                count(table,f"{name}/{setting}/{group}",{m:any(t>0 for t in ev) for m,ev in ports.items()})
            details.append(dict(stream=name,index=index,text=a["text"],group=a["group"],start=start,end=end,
                                isolated=isolated,continuous=continuous))
    return dict(counts=table,records=details,stream_events=streams)


def cohorts():
    table = defaultdict(Counter)
    conflicts = []
    for name in COHORTS:
        astem = f"4090-c161-ambient-history-{name}"
        gstem = f"4090-c161-history-gain-{name}"
        amb, gain = load(astem+"-v1.json"), load(gstem+"-v1.json")
        assert amb["source_sha256"] == gain["reference_sha256"]
        with archive(astem+"-scores-v1.npz",amb["scores_sha256"]) as az, archive(gstem+"-scores-v1.npz",gain["scores_sha256"]) as gz:
            for a,g in zip(amb["records"],gain["records"],strict=True):
                assert (a["phase"],a["index"],a["trace_sha256"]) == (g["phase"],g["index"],g["trace_sha256"])
                key = f"{a['phase']}_{a['index']}"
                n = g["samples"] + 81600
                frames = (n+159)//160-2
                call = np.arange(1,(frames+2)//3+1)
                clock = np.minimum((call+1)*480,n)-81600
                aa = decode(az[key],clock)
                gg = decode(gz[key],clock)
                assert a["ambient"] == aa and g["gain"] == times(gg) and g["c161"] == times(a["c161"])
                pp, suppressed = arbitrate(a["c161"],aa)
                assert a["pair"] == pp and a["suppressed"] == suppressed
                events = dict(c161=g["c161"],gain=times(gg),ambient=times(aa),pair=times(pp))
                source = g["source"]
                label = source.get("target_id") or source.get("phrase_id") or source.get("target") or source.get("group")
                if a["phase"] == "background":
                    counts = table[name+"/background"]
                    counts["samples"] += g["samples"]
                    for model, ev in events.items():
                        counts[model] += sum(t>0 for t in ev)
                elif name == "continuous":
                    for span in g["timeline"]:
                        group = span.get("group",span["kind"])
                        count(table,f"{name}/{a['index']}/{group}",{m:any(span["start"]<t<=span["end"] for t in ev) for m,ev in events.items()})
                else:
                    group = "target" if label in ("hey_nyx","nyx") else "other"
                    count(table,f"{name}/{group}",{m:any(t>0 for t in ev) for m,ev in events.items()})
                    if group == "other" and any(events.values()):
                        conflicts.append(dict(cohort=name,index=g["index"],label=label,text=source.get("text"),events=events))
    return dict(counts=table,other_hits=conflicts)


if __name__ == "__main__":
    result = dict(context=context(),cohorts=cohorts())
    path = RESULTS/"4090-c161-history-paths-review-v1.json"
    with path.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
