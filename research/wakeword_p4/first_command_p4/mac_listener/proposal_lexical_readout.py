"""Replay previously packed Q8 keyword searches at every frozen music-proposal clock."""
import argparse
from pathlib import Path

import music_proposal_replay as base
import train_positive_readout as positive
from check_c161_pocketsphinx import confirmations
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from score_c161_piper_spellings import decoders
from score_proposal_readout import GROUPS

STEM = "4090-c161-proposal-lexical"


def main(groups):
    readers = decoders()
    # The second decoder uses the training-only name-contrast state weights.
    readers = {k:readers[k] for k in ("q8", "name-contrast")}
    assert sha(MODEL) == MODEL_SHA
    packs = [RESULTS / "4090-c161-lexical-p4-q8-packing.json", RESULTS / "4090-c161-q8-name-contrast-fit.json"]
    for group in groups:
        source = RESULTS / f"{positive.STEM}-{group}.json"
        reader, identity = base.scoring.word.load_reader() if group in ("fresh", "acoustics", "matched", "background") else (None, None)
        rows = []
        for (meta, pcm, c161, _, _), ref in zip(base.sources(group), read(source)["records"], strict=True):
            assert base.poll.digest(pcm) == ref["pcm_sha256"] and c161 == ref["c161"]
            proposals = [t for t in ref["proposals"] if 0 < t <= len(pcm)]
            neural = {c["proposal"]:c for c in ref["checks"]}
            checks = {m:confirmations(d, cmn, pcm, proposals) for m,(d,cmn) in readers.items()}
            routes = {}
            for m, found in checks.items():
                accepted = [c for c in found if c["confirmed"] is not None]
                routes[m] = base.scoring.merged(c161, [c["confirmed"] for c in accepted])
                joint = [max(c["confirmed"], neural[c["proposal"]]["end"]) for c in accepted
                         if c["proposal"] in neural and neural[c["proposal"]]["codes"]["trigger"][0] >= 0]
                routes[m+"_and_neural"] = base.scoring.merged(c161, joint)
            row = dict(index=ref["index"], samples=len(pcm), pcm_sha256=ref["pcm_sha256"], c161=c161, proposals=proposals,
                       checks=checks, routes=routes, source=ref.get("source"))
            if group in ("matched", "acoustics"):
                s = ref["source"]
                captures = {}
                for events in routes.values():
                    if events and s["cue_at"] <= events[0] < s["command_at"] and s.get("role","ordinary") in ("ordinary","coded","willow"):
                        begin = str(max(0, ((events[0]+255)//256*256)//160*160-32000))
                        if begin not in captures:
                            captures[begin] = ref["captures"].get(begin) or base.scoring.recognition(reader,pcm[int(begin):],s["expected"])
                row["captures"] = captures
            elif group == "fresh":
                row.update(base.poll.annotate(meta["source"], pcm, routes, [], [], reader))
            elif group == "background":
                row["key"] = ref["key"]
                rows_at = {c["sample"]:c for c in ref["context"]}
                context = []
                for t in sorted(set().union(*routes.values())):
                    if t in rows_at:
                        context.append(rows_at[t])
                    else:
                        lo, hi = max(0,t-80000), min(len(pcm),t+48000)
                        context.append(dict(sample=t,begin=lo,end=hi,asr=base.study.diagnostic.scene.transcripts(reader,pcm[lo:hi])))
                row["context"] = context
            rows.append(row)
        save(RESULTS/f"{STEM}-{group}.json",dict(records=rows, source_sha256=sha(source), c161_sha256=MODEL_SHA,
             decoder_reports={p.name:sha(p) for p in packs}, script_sha256=sha(Path(__file__)), asr=identity,
             policy="At every positive music-proposer clock read 1.5 s history and at most 300 ms later audio. "
             "Use the two fixed Q8 decoders at 1e-15 with the original end tolerance and feature resets. "
             "Notify either on an accepted lexical match, or when that same proposal has both lexical and "
             "trigger-trained neural confirmation. Keep C161 independent with shared native cooldown. "
             "Times measure audio availability, and exclude decoder compute time."))
        print(group, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("groups", nargs="+", choices=GROUPS)
    main(p.parse_args().groups)
