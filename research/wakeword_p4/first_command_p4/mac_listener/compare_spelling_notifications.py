"""Score whole-stream notifications and first-command captures for spelling and five-class confirmations."""
from collections import Counter
from pathlib import Path

import score_spelling_gap as gap
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

prior, study = gap.prior, gap.study


def main():
    inputs = {}

    def load(path):
        inputs[path.name] = sha(path)
        return read(path)

    replay = load(RESULTS / f"{gap.STEM}-replay.json")
    config = load(gap.FIT)
    assert replay["fit_sha256"] == sha(gap.FIT)
    captures = load(RESULTS / f"{prior.STEM}-commands.json")
    reader, identity = study.base.scoring.word.load_reader()
    assert identity == captures["asr"]
    groups, summaries = {}, {}
    for group in study.GROUPS:
        old, timed, ctc = [load(RESULTS / f"{stem}-{group}.json")["records"] for stem in
                           (prior.STEM, study.STEM, prior.fit.reference.STEM)]
        scene = group in ("matched", "acoustics")
        audio = iter(study.base.sources(group)) if scene else None
        rows, counts = [], {}
        for r, t, c, g in zip(old, timed, ctc, replay["records"][group], strict=True):
            assert r["pcm_sha256"] == t["pcm_sha256"] == c["pcm_sha256"] == g["pcm_sha256"]
            assert r["c161"] == t["c161"] == c["c161"]
            routes = dict(c161=r["c161"], paths=r["routes"]["paths"], **t["routes"], gap=g["events"])
            both, either = [], []
            complete = [v for v in c["checks"] if "verifier_end" in v]
            for s, d in zip(g["checks"], complete, strict=True):
                assert s["end"] == d["verifier_end"]
                if d["verifier_codes"][0] < 0:
                    continue
                if s["gap"] <= config["max_gap"] and s["score"] >= config["threshold"]:
                    both.append(s["end"])
                if d["q8"] is not None:
                    either.append(max(s["end"], d["q8"]))
            routes["gap_and_five"] = study.base.scoring.merged(r["c161"], both)
            routes["gap_or_q8_with_five"] = study.base.scoring.merged(r["c161"], both + either)
            source = r["source"]
            row = dict(index=r["index"], pcm_sha256=r["pcm_sha256"], events=routes)
            if scene:
                meta, pcm, c161, _, _ = next(audio)
                assert meta == source and study.base.poll.digest(pcm) == r["pcm_sha256"] and c161 == r["c161"]
                archive = captures["records"][group][r["index"]]
                assert archive["pcm_sha256"] == r["pcm_sha256"] and archive["events"] == routes["paths"]
                cache = {archive["begin"]: archive["asr"]} if "asr" in archive else {}
                row["scene"] = {}
                if group == "matched":
                    row["official_willow"] = archive["official_willow"]
            for k, events in routes.items():
                total = counts.setdefault(k, Counter())
                if scene:
                    kind = source["role"] if group == "matched" else source["acoustic"]
                    timely = bool(events and source["cue_at"] <= events[0] < source["command_at"])
                    total[f"{kind}.trials"] += 1
                    total[f"{kind}.timely"] += timely
                    if timely and source.get("role", "ordinary") in ("ordinary", "coded", "willow"):
                        begin = max(0, ((events[0] + 255) // 256 * 256) // 160 * 160 - 32000)
                        if begin not in cache:
                            cache[begin] = study.base.scoring.recognition(reader, pcm[begin:], source["expected"])
                        capture = cache[begin]
                        total[f"{kind}.complete_suffix"] += capture["complete_suffix"]
                        row["scene"][k] = dict(begin=begin, complete_suffix=capture["complete_suffix"])
                elif group in ("context", "daily", "bot", "fresh"):
                    spans = r["spans"] if group == "fresh" else source["spans"]
                    outside = set(events)
                    for span in spans:
                        stop = span["end"] + (6400 if group == "fresh" else 0)
                        hits = [t for t in events if span["start"] <= t < stop]
                        outside.difference_update(hits)
                        label = span["address"]
                        total[f"{label}.trials"] += 1
                        total[f"{label}.hit"] += bool(hits)
                    total["outside"] += len(outside)
                else:
                    total["events"] += len(events)
                    total["triggered_records"] += bool(events)
                    total["samples"] += r["samples"]
            if scene:
                row["captures"] = cache
            rows.append(row)
        if scene:
            assert next(audio, None) is None
        groups[group], summaries[group] = rows, counts
        print(group, flush=True)
    save(RESULTS / f"{gap.STEM}-comparison.json", dict(summary=summaries, records=groups, inputs=inputs, asr=identity,
         script_sha256=sha(Path(__file__)), policy="Read each full proposal at its original availability time. "
         "Five-class accepts signed Nyx code >=0. Gap or Q8 with five-class uses the first available joint "
         "confirmation at a proposal and all routes merge C161 before cooldown. Strict utterance boundaries, "
         "except for the existing 400 ms fresh challenge grace period. Misses count against all 16 commands."))


if __name__ == "__main__":
    main()
