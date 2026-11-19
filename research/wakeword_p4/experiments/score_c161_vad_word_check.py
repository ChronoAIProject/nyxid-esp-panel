"""Freeze a fresh matched Hi/Hey name challenge and replay the trained extra-word check."""
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import check_c161_piper_spellings as piper
import check_c161_vad_pronunciations as onset
from check_c161_event_transcripts_v1 import choose
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_c161_lexical_echo import infer
from score_c161_piper_spellings import annotate, decoders
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_vad_word_check import FIT, WEIGHTS, STEM, WORK, segment
from train_c161_word_window_v1 import WordWindow, bands

PLAN = RESULTS / f"{STEM}-fresh-plan.json"
ARCHIVE = RESULTS / f"{STEM}-fresh-pcm.zip"
REFERENCE = RESULTS / f"{onset.STEM}.json"


def generate():
    assert sha(WEIGHTS) == read(FIT)["weights_sha256"]
    folder = WORK / "fresh"
    generated = WORK / "piper-raw-plan.json"
    with patch.multiple(piper, PLAN=generated, ARCHIVE=ARCHIVE, WORK=folder,
                        ADDRESSES=("Hye Nyx", "Hey Nyx", "Hey Willow", "Hi Nick", "Hey Nick", "Hi Max", "Hi Chris", "Hi Nikki", "Hey Mick"),
                        CARRIERS=("set the reading lamp to low.", "open the side gate."),
                        DAILY=(("The high notes sounded clear.", "Nick makes lunch on Fridays."),
                               ("A nice mix works well here.", "Please pass me the next one."))):
        piper.generate()
    spec = read(generated)
    spec.update(trained_weights_sha256=sha(WEIGHTS), training_report_sha256=sha(FIT), challenge_script_sha256=sha(Path(__file__)),
                scoring="Host audio-clock development, original dual name-contrast path plus VAD dual-pronunciation search with/without trained segment check. Save complete first-command ASR and each name's result.")
    save(PLAN, spec)


def fresh():
    spec = read(PLAN)
    assert sha(ARCHIVE) == spec["archive_sha256"] and sha(WEIGHTS) == spec["trained_weights_sha256"]
    original = decoders()["name-contrast"]
    paired, model = onset.make_model()
    assert model["files"] == read(REFERENCE)["model"]["files"]
    d = paired["both"]
    cmn = d.get_cmn()
    mid = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"][0]
    assert sha(RESULTS / mid["path"]) == mid["sha256"]
    for i, src in enumerate(spec["records"]):
        pcm = load_wav(src)
        baseline = infer(pcm, RESULTS/mid["path"], *original)["events"]
        windows = [dict(onset=t, start=a, end=b, matches={"both":onset.match(d, cmn, pcm, t, a, b)})
                   for t,a,b in onset.windows(pcm)]
        raw = [e["sample"] for e in baseline["both_proposals"]]
        yield "fresh", str(i), pcm, raw, windows, src, baseline["original"], baseline["expanded"]


def previous():
    prior = {(r["cohort"],r["key"]):r for r in read(REFERENCE)["records"]}
    piper_spec = read(piper.PLAN)["records"]
    piper_eval = read(RESULTS/f"{piper.STEM}-evaluation.json")["records"]
    for group,key,pcm,base,spans in onset.inputs():
        r = prior.pop((group,key))
        assert digest(pcm) == r["pcm_sha256"] and base == r["events"]["base"]
        if group == "piper":
            source = piper_spec[int(key)]
            native, expanded = [piper_eval[int(key)][n] for n in ("native", "expanded")]
        else:
            source = dict(timeline=[dict(address=s["group"], **s) for s in spans])
            native, expanded = [], []
        yield group, key, pcm, base, r["windows"], source, native, expanded
    assert not prior


def main():
    torch.set_num_threads(1)
    assert sha(WEIGHTS) == read(FIT)["weights_sha256"]
    net = WordWindow().cuda().eval()
    with np.load(WEIGHTS) as f:
        net.load_state_dict({k:torch.from_numpy(f[k]) for k in f})
    reader, identity = load_reader()
    records = []
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for cohort in (previous, fresh):
        for group, key, pcm, base, windows, src, native, expanded in cohort():
            matches = [m for w in windows for m in w["matches"]["both"]]
            checks = []
            if matches:
                feature, clock = bands(pcm)
                x = np.stack([segment(feature, clock, m["start"], m["end"], m["available"]) for m in matches])
                with torch.inference_mode():
                    scores = net(torch.from_numpy(x).cuda().float()/255)[:,0].cpu().tolist()
                checks = [dict(**m, logit=v) for m,v in zip(matches,scores,strict=True)]
            def merged(hits):
                return [e["sample"] for e in choose(base, hits)]
            routes = dict(base=base, onset=merged([m["available"] for m in checks]),
                          trained=merged([m["available"] for m in checks if m["logit"] >= 0]))
            if group in ("piper","fresh"):
                scored = annotate(src, pcm, routes, native, expanded, reader)
            else:
                rows = []
                outside = {n:set(v) for n,v in routes.items()}
                for span in src["timeline"]:
                    items = {}
                    for n,v in routes.items():
                        hits = [t for t in v if span["start"] <= t < span["end"]]
                        items[n] = dict(hits=hits)
                        outside[n].difference_update(hits)
                    rows.append(dict(address=span["address"], text=span["text"], routes=items))
                scored = dict(rows=rows, outside={n:sorted(v) for n,v in outside.items()})
            for row in scored["rows"]:
                summary = totals[group][row["address"]]
                summary["trials"] += 1
                for n,v in row["routes"].items():
                    summary[n] += bool(v["hits"])
                    if "exact_command" in v:
                        summary[n+"_complete"] += v["exact_command"]
            records.append(dict(cohort=group, key=key, pcm_sha256=digest(pcm), candidate_checks=checks, events=routes, **scored))
    save(RESULTS/f"{STEM}-evaluation.json", dict(totals=totals, fit_sha256=sha(FIT), prior_sha256=sha(REFERENCE),
         plan_sha256=sha(PLAN), script_sha256=sha(Path(__file__)), asr=identity, records=records))


if __name__ == "__main__":
    import sys
    {"generate":generate, "score":main}[sys.argv[1]]()
