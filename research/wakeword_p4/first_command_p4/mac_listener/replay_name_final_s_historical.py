"""Compare the frozen boundary classifier on the complete legacy, Jenny and Joe speech cohorts."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np

import replay_spelling_gap_dialogues as dialogue
import train_name_final_s as fit
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = fit.base


def main(group):
    assert group in ("primed", "jenny", "joe")
    source = RESULTS / f"4090-c161-history-gain-{group}-v1.json"
    config = read(fit.FIT)
    weights = {}
    for name, meta in config["models"].items():
        archive = RESULTS / meta["file"]
        assert meta["sha256"] == sha(archive)
        with np.load(archive) as saved:
            weights[name] = dict(saved)
    detector = dialogue.Detector()
    reader, identity = dialogue.piper.load_reader()
    rows, counts = [], {}
    for ref in read(source)["records"]:
        if ref["phase"] != "synthetic":
            continue
        src = ref["source"]
        pcm = dialogue.piper.load_wav(src)
        routes, checks = detector.replay(pcm)
        assert ref["c161"] == routes["c161"] and base.poll.digest(pcm) == src["pcm_sha256"]
        accepted = {"parent": [], "veto": []}
        raw = None
        for c in checks:
            if "end" not in c or c["five"][0] < 0:
                continue
            end = c["end"]
            clocks = []
            if c["path"]["interword_blank_frames"] <= detector.config["max_gap"] and c["spelling"]["score"] >= detector.config["threshold"]:
                clocks.append(end)
            if c["q8"]["confirmed"] is not None:
                clocks.append(max(end, c["q8"]["confirmed"]))
            if not clocks:
                continue
            if raw is None:
                raw, _, _ = base.native_inputs(detector.native, pcm)
            patch = np.rint(fit.prior.fit.room.window(raw, end).clip(0,666)*(255./666)).astype(np.uint8)
            scores = {k:int(fit.integer_scores(fit.features(patch), w).max()) for k,w in weights.items()}
            c["s_scores"] = scores
            accepted["parent"] += clocks
            if scores["any_s"] < config["models"]["any_s"]["threshold"] or scores["name_s"] >= config["models"]["name_s"]["threshold"]:
                accepted["veto"] += clocks
        assert base.scoring.merged(routes["c161"], accepted["parent"]) == routes["gap_or_q8_with_five"]
        routes["veto"] = base.scoring.merged(routes["c161"], accepted["veto"])
        label = src.get("target_id", src.get("target"))
        target = label == "hey_nyx"
        command = src["text"].partition(",")[2].strip() if target else ""
        captured = {}
        for k, events in routes.items():
            tally = counts.setdefault(k, Counter())
            tally[f"{label}.trials"] += 1
            tally[f"{label}.hit"] += bool(events)
            if command:
                tally["command.trials"] += 1
                tally["command.hit"] += bool(events)
                if events:
                    start = max(0, (events[0]+159)//160*160 - 32000)
                    if start not in captured:
                        captured[start] = base.scoring.recognition(reader, pcm[start:], command)
                    tally["command.complete_suffix"] += captured[start]["complete_suffix"]
        rows.append(dict(index=ref["index"], pcm_sha256=src["pcm_sha256"], source_id=src["source_id"], label=label,
                         text=src["text"], variant=src["variant"], events=routes, checks=checks, captures=captured))
        if len(rows)%100 == 0:
            print(group,len(rows),flush=True)
    assert len(rows) == {"primed":360, "jenny":396, "joe":360}[group]
    save(RESULTS/f"{fit.STEM}-{group}.json",dict(records=rows,summary=counts, asr=identity,source_sha256=sha(source),
         script_sha256=sha(Path(__file__)),fit_sha256=sha(fit.FIT),identity=detector.identity,
         policy="Every archived speech file and source hash, including all negative and bare controls. "
         "One intact primed run per file; original C161 clock equality required. Hits use the container as in "
         "the historical report. Transcribe 2 s pre-roll after each first returned notification for commands. "
         "No added background exposure or assumption that variants are independent speakers."))


if __name__ == "__main__":
    for cohort in sys.argv[1:]:
        main(cohort)
