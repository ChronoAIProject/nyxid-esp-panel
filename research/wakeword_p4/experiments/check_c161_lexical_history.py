"""Select the length of one Q8 verification window on training proposals, then replay development sources."""
import argparse
from collections import defaultdict
from fractions import Fraction

import numpy as np
import soundfile as sf
from pocketsphinx import Decoder

from check_c161_compact_lexical import WORK, sources as dev_sources
from check_c161_event_transcripts_v1 import choose
from check_c161_midlevel_pocketsphinx import STEM
from check_c161_pocketsphinx import search
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_word_window_v1 import sources as training_sources

PLAN = RESULTS/"4090-c161-proposal-windows-training-v1.json"
FIT = RESULTS/"4090-c161-lexical-history-fit.json"
PACK = RESULTS/"4090-c161-lexical-p4-q8-packing.json"
# Integral 30 ms chunks keep the physical polling phase fixed.
HISTORIES = (24000, 14400, 19200, 33600)


def setup():
    files = WORK/"q8/files"
    assert all(sha(files/n)==h for n,h in read(PACK)["files"].items())
    decoder = Decoder(hmm=str(files/"am"),dict=str(files/"two.dict"),keyphrase="hey nicks",
                      kws_threshold=1e-15,loglevel="ERROR",lm=None)
    return decoder, decoder.get_cmn()


def window(d, cmn, pcm, t, past):
    start, end = max(0,t-past), min(len(pcm),t+4800)
    matches = search(d,cmn,pcm[start:end],start)
    times = [max(t,m["returned"]) for m in matches if t-7200<=m["end"]<=end]
    return dict(proposal=t,start=start,end=end,matches=matches,confirmed=min(times,default=None))


def fit():
    d, cmn = setup()
    frozen = {r["key"]:r for r in read(PLAN)["training"]}
    rows = []
    for src, pcm in training_sources():
        row = frozen.pop(src["key"])
        assert digest(pcm)==row["pcm_sha256"] and src["positive"]==row["positive"]
        if not row["ticks"]:
            continue
        # The frozen training sampler defined these proposals on 0.5 s of leading silence.
        pcm = np.pad(pcm,(8000,8000))
        tests = {}
        for past in HISTORIES:
            tests[str(past)] = [window(d,cmn,pcm,8000+t,past)["confirmed"] for t in row["ticks"]]
        rows.append(dict(key=src["key"],positive=src["positive"],ticks=row["ticks"], tests=tests))
    assert not frozen
    rank, counts = [], {}
    for h in HISTORIES:
        pos = [any(t is not None for t in r["tests"][str(h)]) for r in rows if r["positive"]]
        neg = [all(t is None for t in r["tests"][str(h)]) for r in rows if not r["positive"]]
        rank.append(Fraction(sum(pos),len(pos))+Fraction(sum(neg),len(neg)))
        counts[h]=dict(positive=[sum(pos),len(pos)],negative=[sum(neg),len(neg)])
    best = HISTORIES[max(range(len(rank)),key=rank.__getitem__)]
    save(FIT,dict(proposals_sha256=sha(PLAN),model_manifest_sha256=sha(PACK), histories=HISTORIES,
                  selected=best,counts=counts,records=rows))
    print(best,counts,flush=True)


def evaluate():
    d, cmn = setup()
    selected = read(FIT)["selected"]
    full_path = RESULTS/f"{STEM}-v1.json"
    bg_path = RESULTS/f"{STEM}-background-v1.json"
    originals = {(r["cohort"],r["key"]):r for r in read(full_path)["records"]}
    all_rows=[]
    def check(row, pcm, group, key):
        checked = {str(h):[window(d,cmn,pcm,c["proposal"],h) for c in row["checked"]]
                   for h in dict.fromkeys((24000,selected))}
        events = {h:choose(row["baseline"],[c["confirmed"] for c in checks if c["confirmed"] is not None])
                  for h,checks in checked.items()}
        assert events["24000"] == row["events"]
        all_rows.append(dict(cohort=group,key=key,pcm_sha256=row["pcm_sha256"],events=events,checks=checked))
    for group,key,pcm in dev_sources():
        row=originals.pop((group,key))
        assert digest(pcm)==row["pcm_sha256"]
        check(row,pcm,group,key)
    assert not originals
    lookup = {}
    for group in ("melo","primed","jenny"):
        for row in read(RESULTS/f"4090-c161-history-gain-{group}-v1.json")["records"]:
            if row["phase"]=="background":
                lookup[group,f"{group}_{row['index']}"]=row["source"]
    for row in read(bg_path)["records"]:
        group,key=row["cohort"],row["key"]
        src=lookup.pop((group,key))
        if row["checked"]:
            pcm,rate=sf.read(src.get("wav_path") or src["path"],dtype="int16")
            assert rate==16000 and digest(pcm)==row["pcm_sha256"]
        else:
            pcm=None
        check(row,pcm,group,key)
    assert not lookup
    save(RESULTS/"4090-c161-lexical-history-evaluation.json",dict(fit_sha256=sha(FIT),
         source_sha256=sha(full_path),background_sha256=sha(bg_path),records=all_rows))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("fit","evaluate"))
    {"fit":fit,"evaluate":evaluate}[parser.parse_args().action]()
