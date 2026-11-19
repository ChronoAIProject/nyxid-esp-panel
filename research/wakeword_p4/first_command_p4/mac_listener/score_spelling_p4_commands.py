"""Measure command captures and localize remaining misses at the fixed proposal clocks."""
from collections import Counter, defaultdict
from pathlib import Path

import check_spelling_p4 as probe
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = probe.base


def outcome(events, meta, pcm, reader, cache):
    timely = bool(events and meta["cue_at"] <= events[0] < meta["command_at"])
    out = dict(events=events, timely=timely)
    if timely and meta.get("role","ordinary") in ("ordinary","coded","willow"):
        begin = max(0, ((events[0]+255)//256*256)//160*160-32000)
        if begin not in cache:
            cache[begin] = base.scoring.recognition(reader, pcm[begin:], meta["expected"])
        out.update(begin=begin, complete_suffix=cache[begin]["complete_suffix"])
    return out


def opportunity(stream, windows, meta, result):
    lo, hi = meta["cue_at"], meta["command_at"]
    eligible = [w for w in windows if lo <= w["end"] < hi]
    available = [w for w in windows if lo <= w["complete_sample"] < hi]
    if result["timely"]:
        reason = "timely"
    elif result["events"] and result["events"][0] < lo:
        reason = "early_first_event"
    elif not available:
        reason = "worker_late" if eligible else "no_window_before_command"
    elif not any(w["device"]["capped"] for w in available):
        reason = "spelling_rejected"
    else:
        raise AssertionError((stream, meta, result))
    return dict(reason=reason, raw_windows_in_cue=len(eligible), finished_windows_in_cue=len(available),
                raw_margin_samples=[hi-w["end"] for w in eligible],
                finished_margin_samples=[hi-w["complete_sample"] for w in available])


def main():
    measured_path = RESULTS / (probe.STEM+".json")
    measured = read(measured_path)
    comparison_path = RESULTS / (probe.gap.STEM+"-comparison.json")
    archive = read(comparison_path)
    reader, identity = base.scoring.word.load_reader()
    assert identity == archive["asr"]
    records, summary = {}, {}
    for group in ("acoustics","matched"):
        counts = defaultdict(lambda: defaultdict(Counter))
        records[group] = []
        streams = [s for s in measured["streams"] if s["group"] == group]
        for (meta,pcm,c161,_,_), s, old in zip(base.sources(group), streams, archive["records"][group], strict=True):
            assert s["index"] == old["index"] and base.poll.digest(pcm) == s["pcm_sha256"] == old["pcm_sha256"]
            assert s["host"]["paths"] == old["events"]["paths"] and s["host"]["capped"] == old["events"]["gap"]
            assert c161 == s["c161"] == old["events"]["c161"]
            cache = {int(k):v for k,v in old["captures"].items()}
            cached = set(cache)
            clocks = dict(c161=c161, host_paths=s["host"]["paths"], host_gap=s["host"]["capped"],
                          p4_paths=s["delayed"]["paths"], p4_gap=s["delayed"]["capped"])
            outcomes = {k:outcome(ts,meta,pcm,reader,cache) for k,ts in clocks.items()}
            if group == "matched":
                willow = old["official_willow"]
                official = dict(events=willow["events"], timely=willow["timely"])
                if willow["capture"] is not None:
                    assert official["timely"]
                    official["complete_suffix"] = willow["capture"]["complete_suffix"]
                outcomes["official_willow"] = official
            kind = meta["role"] if group == "matched" else meta["acoustic"]
            for route, result in outcomes.items():
                total = counts[kind][route]
                total["trials"] += 1
                total["timely"] += result["timely"]
                total["timely_complete"] += result.get("complete_suffix",False)
            windows = [measured["records"][i] for i in s["windows"]]
            audit = opportunity(s,windows,meta,outcomes["p4_gap"])
            counts[kind]["p4_gap_outcomes"][audit["reason"]] += 1
            records[group].append(dict(index=s["index"], pcm_sha256=s["pcm_sha256"], source=meta,
                                       routes=outcomes, opportunity=audit,
                                       new_transcripts={k:v for k,v in cache.items() if k not in cached}))
        summary[group] = counts
        print(group, flush=True)
    save(RESULTS / (probe.STEM+"-commands.json"), dict(records=records, summary=summary, asr=identity,
         measurement_sha256=sha(measured_path), comparison_sha256=sha(comparison_path), script_sha256=sha(Path(__file__)),
         policy="First merged event on or after cue onset and strictly before command. Two-second pre-roll "
                "followed by all remaining scene audio. Reuse an archived transcript only for the same PCM "
                "hash and capture start. Official Willow uses the corresponding frozen matched-room recording. "
                "Reasons describe the full-window availability and actual timed decisions for each first event."))


if __name__ == "__main__":
    main()
