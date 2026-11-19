"""Apply the two integer /s/ classifiers at unchanged eligible proposal clocks."""
from pathlib import Path

import numpy as np

import replay_spelling_gap_dialogues as dialogue
import train_name_final_s as fit
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base, prior, gap = fit.base, fit.prior, fit.gap


def old_streams(group, files):
    def load(path):
        files[path.name] = sha(path)
        return read(path)
    joint = load(RESULTS / f"{prior.fit.reference.STEM}-{group}.json")["records"]
    trace = load(RESULTS / f"{gap.STEM}-replay.json")["records"][group]
    old = load(RESULTS / f"{gap.STEM}-comparison.json")["records"][group]
    letters = load(RESULTS / f"{prior.fit.STEM}-{group}.json")["records"]
    for (_, pcm, c161, _, _), j, g, r, l in zip(base.sources(group), joint, trace, old, letters, strict=True):
        assert base.poll.digest(pcm) == j["pcm_sha256"] == g["pcm_sha256"] == r["pcm_sha256"] == l["pcm_sha256"]
        complete = [c for c in j["checks"] if "verifier_end" in c]
        checks = []
        for c, a, b in zip(complete, g["checks"], l["checks"], strict=True):
            assert c["verifier_end"] == a["end"] == b["end"]
            checks.append((a["end"], a["gap"], a["score"], c["q8"], c["verifier_codes"][0], b["patch_sha256"]))
        yield pcm, c161, checks, r["events"]["gap_or_q8_with_five"], l


def new_streams(files):
    path = RESULTS / f"{dialogue.corpus.STEM}-replay.json"
    files[path.name] = sha(path)
    report = read(path)
    assert report["plan_sha256"] == sha(dialogue.corpus.PLAN)
    for s, r in zip(read(dialogue.corpus.PLAN)["records"], report["records"], strict=True):
        pcm = dialogue.piper.load_wav(s)
        assert base.poll.digest(pcm) == r["pcm_sha256"]
        checks = [(c["end"], c["path"]["interword_blank_frames"], c["spelling"]["score"], c["q8"]["confirmed"],
                   c["five"][0], None) for c in r["checks"] if "end" in c]
        yield pcm, r["events"]["c161"], checks, r["events"]["gap_or_q8_with_five"], dict(source=s)


def main():
    spec, conf = read(fit.FIT), read(gap.FIT)
    native = prior.fit.native_frontend()
    reader, identity = base.scoring.word.load_reader()
    weights = {}
    for mode, meta in spec["models"].items():
        file = RESULTS / meta["file"]
        assert sha(file) == meta["sha256"]
        with np.load(file) as w:
            weights[mode] = dict(w)
    result, files = {}, {}
    for group in ("dialogue", *gap.study.GROUPS):
        streams = new_streams(files) if group == "dialogue" else old_streams(group, files)
        rows = []
        for pcm, c161, checks, parent, src in streams:
            allowed = {"parent": [], "veto": [], **{m: [] for m in weights}}
            outputs, raw = [], None
            for end, blanks, spelling, q8, five, expected_patch in checks:
                clocks = []
                if five >= 0:
                    if blanks <= conf["max_gap"] and spelling >= conf["threshold"]:
                        clocks.append(end)
                    if q8 is not None:
                        clocks.append(max(q8, end))
                if not clocks:
                    continue
                allowed["parent"] += clocks
                if raw is None:
                    raw, _, _ = base.native_inputs(native, pcm)
                patch = np.rint(prior.fit.room.window(raw, end).clip(0,666)*(255./666)).astype(np.uint8)
                digest = base.poll.digest(patch)
                assert expected_patch is None or expected_patch == digest
                x = fit.features(patch)
                scores = {m: fit.integer_scores(x, w) for m, w in weights.items()}
                for m, v in scores.items():
                    if v.max() >= spec["models"][m]["threshold"]:
                        allowed[m] += clocks
                if (scores["any_s"].max() < spec["models"]["any_s"]["threshold"] or
                        scores["name_s"].max() >= spec["models"]["name_s"]["threshold"]):
                    allowed["veto"] += clocks
                outputs.append(dict(end=end, confirm=clocks, patch_sha256=digest,
                                    scores={m:int(v.max()) for m,v in scores.items()},
                                    peak_frames={m:int(fit.CENTERS[v.argmax()]) for m,v in scores.items()}))
            routes = {m:base.scoring.merged(c161, ts) for m,ts in allowed.items()}
            assert routes["parent"] == parent
            row = dict(index=len(rows), pcm_sha256=base.poll.digest(pcm), checks=outputs, events=routes)
            if group == "dialogue":
                row.update(dialogue.piper.annotate(src["source"], pcm, routes, [], [], reader))
            elif group in ("matched","acoustics"):
                s = src["source"]
                row["captures"] = {}
                for m, ts in routes.items():
                    if ts and s["cue_at"] <= ts[0] < s["command_at"] and s.get("role", "ordinary") == "ordinary":
                        start = max(0, ((ts[0]+255)//256*256)//160*160 - 32000)
                        if start not in row["captures"]:
                            row["captures"][start] = base.scoring.recognition(reader, pcm[start:], s["expected"])
            rows.append(row)
        result[group] = rows
        print(group, flush=True)
    save(RESULTS / f"{fit.STEM}-replay.json", dict(records=result, files=files, fit_sha256=sha(fit.FIT),
         gap_fit_sha256=sha(gap.FIT), asr=identity, script_sha256=sha(Path(__file__)),
         policy="Only previously eligible auxiliary checks undergo the additional acoustic gate. Freeze both "
         "thresholds and recompute merged C161 notifications, including cooldown. An additional retrospective veto rejects only when any_s passes and name_s fails. Require the ungated "
         "reconstruction to match every archived event. Report all six new dialogues and eight older development groups."))


if __name__ == "__main__":
    main()
