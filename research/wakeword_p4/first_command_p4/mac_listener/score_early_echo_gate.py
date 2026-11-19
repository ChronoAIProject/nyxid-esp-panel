"""Replay fitted early gates on component measurements and the existing background clocks."""
from collections import Counter, defaultdict
from pathlib import Path

import check_early_echo_interference as interference
import fit_early_echo_gate as fit
import score_calibrated_echo_opus as opus

timing, ch = fit.timing, fit.ch
OUT = ch.RESULTS / "4090-c161-early-echo-gate-replay.json"


def routes(row, windows, features, spec, cutoff):
    proposals = row["stages"]["filtered"]["proposals"]
    early = {t+timing.EARLY for t in proposals}
    assert early.isdisjoint(t+timing.STANDARD for t in proposals)
    result = {"standard":timing.schedule(row,windows,(timing.STANDARD,),cutoff),
              "early":timing.schedule(row,windows,(timing.EARLY,timing.STANDARD),cutoff)}
    for name in ("linear","scalar"):
        gated = dict(windows)
        for end in early.intersection(windows):
            if not windows[end]["passes"]:
                continue
            x = features[end]
            score = sum(a*b for a,b in zip(x,spec["weights"],strict=True))
            assert -(1<<31) <= score < (1<<31)
            passed = score >= spec["linear_cutoff"] if name == "linear" else x[3] >= spec["scalar_nyx_cutoff"]
            gated[end] = dict(windows[end], passes=passed)
        result[name] = timing.schedule(row,gated,(timing.EARLY,timing.STANDARD),cutoff)
    return result


def main():
    spec = ch.read(fit.COEFF)
    assert ch.sha(fit.FIT) == spec["fit_sha256"]
    parent, measured = ch.read(timing.names.OUT), ch.read(timing.OUT)
    assert measured["models"] == spec["models"] and ch.sha(timing.names.OUT) == measured["input_sha256"]
    char, five = ch.read(timing.CHAR), ch.read(timing.FIVE)
    assert (ch.sha(timing.CHAR),ch.sha(timing.FIVE)) == (measured["char_sha256"],measured["five_sha256"])
    codes = {w["character_window"]:w for w in five["records"]}
    captures = ch.read(opus.OUT)
    assert ch.sha(timing.OUT) == captures["timing_sha256"]
    views = {r["ordinal"]:r for r in captures["records"]}
    results, summary = [], defaultdict(lambda:defaultdict(Counter))
    for i, (row, previous, stream) in enumerate(zip(parent["records"],measured["records"],char["streams"],strict=True)):
        assert i == previous["ordinal"] == stream["index"] and row["mic_sha256"] == previous["mic_sha256"]
        windows, features = {}, {}
        for j in stream["windows"]:
            w = char["records"][j]
            cost = w["invoke_us"] + w["decode_us"]
            passes = False
            if j in codes:
                device = codes[j]["device"]
                features[w["end"]] = fit.features(w["device"],device)
                cost += codes[j]["invoke_us"]
                passes = device[0] >= spec["models"]["cutoff"]
            windows[w["end"]] = dict(id=j,service_us=cost,passes=passes)
        runs = routes(row,windows,features,spec,parent["cutoff_samples"])
        assert all(runs[k] == previous[k] for k in ("standard","early"))
        item = dict(ordinal=i, role=row["role"], routes={})
        for k,v in runs.items():
            total = summary[row["role"]][k]
            total.update(trials=1,hit=v["hit"],timely=v["before_command"],windows=len(v["jobs"]))
            item["routes"][k] = dict(events=v["events"], jobs=len(v["jobs"]))
            if v["hit"] and row["role"] in opus.TARGETS:
                opened = next(t for t in v["events"] if row["start"]<=t<row["end"])
                first = max(0,(opened+159)//160*160-32000)
                known = [captures["captures"][m["capture"]] for m in views[i]["routes"].values() if m is not None]
                cap = next(c for c in known if c["first"]==first and c["end"]==row["end"])
                complete = cap["lanes"]["residual"]["complete"]
                total.update(complete=complete,timely_complete=v["before_command"] and complete)
                item["routes"][k].update(capture=cap["index"],decoded_sha256=cap["lanes"]["residual"]["decoded_sha256"],
                                       complete=complete)
        results.append(item)
    archive = ch.read(interference.OUT)
    assert archive["models"] == spec["models"]
    material = {}
    for group, records in archive["records"].items():
        material[group] = []
        for prev in records:
            windows, features = {}, {}
            for key,w in prev["checks"].items():
                end = int(key)
                passes = False
                cost = parent["char_us"]
                if w["accepted"]:
                    features[end] = fit.features(w,w["five"])
                    passes = w["five"][0] >= spec["models"]["cutoff"]
                    cost += parent["five_us"]
                windows[end] = dict(id=end,service_us=cost,passes=passes)
            row = dict(start=0,end=prev["samples"],command_at=0,stages=dict(raw=dict(c161=prev["independent"]),
                            filtered=dict(c161=[],proposals=prev["proposals"])))
            runs = routes(row, windows, features, spec, 0)
            assert all(runs[k]["events"] == prev["routes"][k]["events"] for k in ("standard","early"))
            item = dict(index=prev["index"],pcm_sha256=prev["pcm_sha256"],routes={})
            for k,v in runs.items():
                total = summary[group][k]
                total.update(trials=1,events=len(v["events"]),windows=len(v["jobs"]),samples=prev["samples"])
                item["routes"][k] = v["events"]
                if group in ("daily","bot"):
                    for s in prev["source"]["source"]["spans"]:
                        total[s["address"]+"_spans"] += 1
                        total[s["address"]+"_hits"] += any(s["start"]<=t<s["end"]+6400 for t in v["events"])
            material[group].append(item)
    sources = (fit.COEFF,timing.names.OUT,timing.OUT,timing.CHAR,timing.FIVE,opus.OUT,interference.OUT,Path(__file__))
    ch.save(OUT,dict(records=results,interference=material,summary=summary,sources={p.name:ch.sha(p) for p in sources},
            policy="Only the additional 180 ms acceptance can be gated. Frozen 360 ms fallback and independent C161. "
            "Use per-window P4 service time on calibrated residuals and archived P95 time on raw interference "
            "inputs. Integer comparison overhead excluded. For target openings verify the identical capture "
            "start and end in that same source's decoded Opus archive. All comparisons on consumed development data."))


if __name__ == "__main__":
    main()
