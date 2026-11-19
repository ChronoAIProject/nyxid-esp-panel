"""Use pre-proposal continuous speech onsets to condition the auxiliary wake path."""
import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np

import music_proposal_replay as base
import train_positive_readout as relaxed
import train_proposal_readout as strict
from check_c161_vad_pronunciations import windows
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from train_c161_onset_names import WAIT as ADDRESS_WINDOW

STEM = "4090-c161-readout-onsets"


def main(group):
    paths = [RESULTS/f"{fit.STEM}-{group}.json" for fit in (relaxed,strict)]
    positive,negative = (read(path)["records"] for path in paths)
    out = []
    reader, identity = base.scoring.word.load_reader() if group in ("acoustics","matched","fresh") else (None,None)
    for (meta,pcm,c161,_,_),pos,neg in zip(base.sources(group),positive,negative,strict=True):
        assert pos["pcm_sha256"] == neg["pcm_sha256"] == base.poll.digest(pcm) and c161 == pos["c161"] == neg["c161"]
        assert pos["proposals"] == neg["proposals"] and pos["routes"]["parent"] == neg["routes"]["parent"]
        onsets = [t for t,_,_ in windows(pcm)]
        checks = []
        for a,b in zip(pos["checks"],neg["checks"],strict=True):
            assert (a["proposal"],a["end"]) == (b["proposal"],b["end"])
            t = a["proposal"]
            i = bisect_right(onsets,t)
            onset = onsets[i-1] if i else None
            active = onset is not None and t-onset <= ADDRESS_WINDOW
            allow = a["codes"]["trigger"][0]>=0
            older = a["codes"]["parent"][0]>=0
            conservative = b["codes"]["peak"][0]>=0
            checks.append(dict(proposal=t,end=a["end"],last_onset=onset,in_window=active,
                       parent_onset=active and older,trigger_onset=active and allow,
                       switch=allow if active else conservative))
        routes = {k:base.scoring.merged(c161,[c["end"] for c in checks if c[k]])
                  for k in ("parent_onset","trigger_onset","switch")}
        # Hash every onset without storing the many unassociated speech boundaries on long backgrounds.
        row = dict(source=meta, index=len(out), pcm_sha256=pos["pcm_sha256"], samples=len(pcm),
                   onset_count=len(onsets), onset_sha256=base.poll.digest(np.asarray(onsets,"<i4")),
                   c161=c161, checks=checks, routes=routes)
        if group in ("acoustics","matched"):
            captures = {**neg["captures"], **pos["captures"]}
            for ev in routes.values():
                if ev and meta["cue_at"]<=ev[0]<meta["command_at"] and meta.get("role","ordinary") in ("ordinary","coded","willow"):
                    begin = str(max(0, ((ev[0]+255)//256*256)//160*160-32000))
                    if begin not in captures:
                        captures[begin] = base.scoring.recognition(reader,pcm[int(begin):],meta["expected"])
            row["captures"] = captures
        elif group == "fresh":
            row.update(base.poll.annotate(meta["source"],pcm,routes,[],[],reader))
            row["source"] = {k:v for k,v in meta["source"].items() if k != "timeline"}
        elif group == "background":
            contexts = {r["sample"]:r for prev in (pos,neg) for r in prev["context"]}
            events = sorted(set().union(*routes.values()))
            row["background_context"] = [contexts[t] for t in events if t in contexts]
        out.append(row)
    save(RESULTS/f"{STEM}-{group}.json",dict(records=out,source_reports={p.name:sha(p) for p in paths},
        vad_sha256=sha(Path(windows.__code__.co_filename)), script_sha256=sha(Path(__file__)), asr=identity,
        policy="C161 always contributes. Pocketsphinx loose VAD, 30 ms frames, onset following 450 ms unvoiced. "
               "Use the last onset already available at word proposal; recent means within the pre-existing "
               "1.32-second address window. Compare parent or positive trigger verifier only after recent onsets, "
               "and the trigger verifier near an onset with peak-negative verifier otherwise. Same 360 ms and cooldown."))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("groups",nargs="+",choices=("acoustics","matched","fresh","context","daily","bot","music","background"))
    for group in p.parse_args().groups:
        main(group)
