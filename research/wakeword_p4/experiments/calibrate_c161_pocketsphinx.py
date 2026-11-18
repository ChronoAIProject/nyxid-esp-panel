"""Select an English keyphrase threshold on previously admitted training sources only."""
from fractions import Fraction
import sys

import numpy as np

from check_c161_pocketsphinx import STEM, background, run, search, setup
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_word_window_v1 import sources

PLAN = RESULTS / "4090-c161-word-window-training-v1.json"
OUT = RESULTS / f"{STEM}-calibration-v1.json"
EXPONENTS = (5, 10, 15, 20, 25, 30)


def calibrate():
    inventory = {r["key"]: r for r in read(PLAN)["training"]}
    clips = []
    for src, pcm in sources():
        row = inventory.pop(src["key"])
        assert digest(pcm) == row["pcm_sha256"] and src["positive"] == row["positive"]
        group = "target" if src["positive"] else "address" if src["text"].lower().startswith("hey ") else "other"
        clips.append((src, group, np.pad(pcm, (8000,8000))))
    assert not inventory and len(clips) == 918
    reports, order = [], []
    for exponent in EXPONENTS:
        decoder, cmn, _ = setup(10.**(-exponent))
        rows = []
        for src, group, pcm in clips:
            matches = search(decoder, cmn, pcm, -8000)
            if group == "target":
                correct = any(0 <= e["returned"] <= src["end"] + 4800 for e in matches)
            else:
                correct = not matches
            rows.append(dict(key=src["key"], group=group, correct=correct, times=[e["returned"] for e in matches]))
        rates = {}
        counts = {}
        for group in ("target","address","other"):
            subset = [r for r in rows if r["group"] == group]
            counts[group] = [sum(r["correct"] for r in subset), len(subset)]
            rates[group] = Fraction(*counts[group])
        score = rates["target"]/2 + (rates["address"]+rates["other"])/4
        order.append((score, rates["other"], rates["target"]))
        reports.append(dict(exponent=exponent, counts=counts, score=float(score), records=rows))
        print(exponent, counts, float(score), flush=True)
    selected = max(range(len(order)), key=order.__getitem__)
    save(OUT, dict(training_sha256=sha(PLAN), selected_exponent=EXPONENTS[selected], measurements=reports))


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        calibrate()
    else:
        {"speech":run, "background":background}[sys.argv[1]](10.**(-read(OUT)["selected_exponent"]), STEM+"-calibrated")
