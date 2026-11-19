"""Calibrate a limit on the blank run between HEY/HAY and the name in a winning spelling."""
import argparse
from pathlib import Path

import numpy as np

import trace_spelling_conflicts as diagnostic
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

prior = diagnostic.prior
study = diagnostic.study
STEM = "4090-c161-spelling-gap"
FIT = RESULTS / f"{STEM}-training.json"


def calibrate():
    vm, vocab = prior.model(), read(prior.WAV2VEC / "vocab.json")
    windows, _, sources, _ = prior.fit.training_windows()
    old = read(prior.CAL)
    assert sources == read(prior.fit.FIT)["records"]
    rows = []
    for patch, s, c in zip(windows, sources, old["records"], strict=True):
        assert (s["key"], s["kind"], s["label"]) == (c["key"], c["kind"], c["label"])
        path = diagnostic.trace(prior.outputs(vm, patch), c["target"], vocab)
        assert path["normalized"] == c["target_value"]
        rows.append(dict(key=s["key"], view=s["kind"], label=s["label"], score=c["score"],
                         gap=path["interword_blank_frames"], patch_sha256=s["patch_sha256"]))
    # Preserve at least 95% of the previously accepted positive training alignments.
    positive = [r for r in rows if r["label"] == 0]
    admitted = [r for r in positive if r["score"] >= old["threshold"]]
    limit = int(np.quantile([r["gap"] for r in admitted], .95, method="higher"))
    eligible = sorted((r["score"] for r in positive if r["gap"] <= limit), reverse=True)
    threshold = eligible[len(admitted) - 1]
    save(FIT, dict(records=rows, max_gap=limit, threshold=threshold, old_count=len(admitted),
                   new_count=sum(r["gap"] <= limit and r["score"] >= threshold for r in positive),
                   old_calibration_sha256=sha(prior.CAL), script_sha256=sha(Path(__file__)),
                   trace_script_sha256=sha(Path(diagnostic.__file__)),
                   policy="Upper 95th percentile (higher) of inter-word blanks in old admitted positive training "
                          "views. Lower the original score cutoff just enough to recover the old target view count; "
                          "include all ties. Gate the old winning target alignment, with no alternative alignment search."))


def evaluate():
    config = read(FIT)
    assert config["script_sha256"] == sha(Path(__file__))
    assert config["trace_script_sha256"] == sha(Path(diagnostic.__file__))
    vm, vocab, native = prior.model(), read(prior.WAV2VEC / "vocab.json"), prior.fit.native_frontend()
    inputs, groups = {}, {}
    for group in study.GROUPS:
        files = [RESULTS / f"{stem}-{group}.json" for stem in (prior.STEM, prior.fit.STEM)]
        inputs.update({p.name: sha(p) for p in files})
        rows = []
        for (_, pcm, c161, _, _), old, letters in zip(study.base.sources(group), *(read(p)["records"] for p in files), strict=True):
            assert study.base.poll.digest(pcm) == old["pcm_sha256"] == letters["pcm_sha256"]
            assert c161 == old["c161"] == letters["c161"]
            admitted, checks = [], []
            if old["checks"]:
                raw, _, _ = study.base.native_inputs(native, pcm)
            for c, d in zip(old["checks"], letters["checks"], strict=True):
                assert c["end"] == d["end"]
                patch = np.rint(prior.fit.room.window(raw, c["end"]).clip(0, 666) * (255. / 666)).astype(np.uint8)
                codes = prior.outputs(vm, patch)
                assert study.base.poll.digest(codes) == d["models"]["student"]["output_codes_sha256"]
                path = diagnostic.trace(codes, c["target"], vocab)
                assert path["normalized"] == c["target_value"]
                gap = path["interword_blank_frames"]
                if gap <= config["max_gap"] and c["score"] >= config["threshold"]:
                    admitted.append(c["end"])
                checks.append(dict(end=c["end"], gap=gap, score=c["score"]))
            rows.append(dict(index=old["index"], pcm_sha256=old["pcm_sha256"], checks=checks,
                             events=study.base.scoring.merged(c161, admitted)))
        groups[group] = rows
        print(group, flush=True)
    save(RESULTS / f"{STEM}-replay.json", dict(records=groups, inputs=inputs, fit_sha256=sha(FIT),
                                               script_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["calibrate", "evaluate"])
    {"calibrate": calibrate, "evaluate": evaluate}[parser.parse_args().action]()
