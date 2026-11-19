"""Calibrate one signed Nyx cutoff against the preceding five-class head's training false alarms."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import replay_early_proposal_readout as study
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

training = study.training
FIT = RESULTS / (training.STEM + "-cutoff.json")
GROUPS = ("synthetic_other", "speech", "music", "daily")


def main():
    models, ids = study.models()
    native = training.calibration.prior.fit.native_frontend()
    rows = []
    counts = defaultdict(Counter)
    for (source, pcm), old in zip(training.sources(), read(training.FIT)["records"], strict=True):
        assert source["key"] == old["key"] and study.base.poll.digest(pcm) == old["pcm_sha256"]
        if not old["training"]:
            continue
        raw = study.base.native_inputs(native, pcm)[0]
        for window in old["training"]:
            patch = np.rint(training.room.window(raw, window["end"]).clip(0, 666) * (255 / 666)).astype(np.uint8)
            assert study.base.poll.digest(patch) == window["patch_sha256"]
            codes = {k:int(model.invoke(patch)[0]) for k, model in models.items()}
            group = source["mode"]
            if group in ("dry", "room"):
                group = "target" if source["positive"] else ("synthetic_other" if window["label"] == 4 else "name")
            counts[group][(codes["baseline"], codes["trained"])] += 1
            rows.append(dict(key=source["key"], mode=source["mode"], end=window["end"], group=group,
                             patch_sha256=window["patch_sha256"], **codes))
    cutoffs = {}
    for group in GROUPS:
        scores = counts[group]
        budget = sum(n for (b, _), n in scores.items() if b >= 0)
        cutoffs[group] = next(c for c in range(-128, 129) if
                              sum(n for (_, t), n in scores.items() if t >= c) <= budget)
    selected = max(cutoffs.values())
    summary = {g: dict(windows=sum(scores.values()),
                       baseline=sum(n for (b, _), n in scores.items() if b >= 0),
                       trained=sum(n for (_, t), n in scores.items() if t >= 0),
                       calibrated=sum(n for (_, t), n in scores.items() if t >= selected))
               for g, scores in counts.items()}
    save(FIT, dict(records=rows, summary=summary, models=ids, cutoffs=cutoffs, cutoff=selected,
         fit_sha256=sha(training.FIT), script_sha256=sha(Path(__file__)), policy="Minimum signed Nyx score "
         "whose admitted negative training proposals do not exceed the parent count in each of four "
         "strata: synthetic other, LibriSpeech train-clean, music, and daily training. Named confusions "
         "are reported separately. Calibration uses only verified training windows, before spelling."))


if __name__ == "__main__":
    main()
