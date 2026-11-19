"""Compare both trained front ends with the archived parent and frozen spelling gate."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import check_spelling_p4 as probe
import train_proposal_deadlines as fit
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from score_spelling_p4_commands import outcome

base, prior, gap = probe.base, probe.prior, probe.gap


def sources(group):
    if group in gap.study.GROUPS:
        old = read(RESULTS/f"{prior.STEM}-{group}.json")["records"]
        for (meta,pcm,c161,_,_), r in zip(base.sources(group),old,strict=True):
            assert base.poll.digest(pcm) == r["pcm_sha256"] and r["c161"] == c161
            spans = r["source"]["spans"] if group in ("context","daily","bot") else r["spans"]
            yield r["index"], pcm, c161, meta if group in ("matched","acoustics") else {}, spans
    else:
        joe = group == "joe"
        plan = RESULTS/(("4090-c161-history-gain-joe-v1" if joe else probe.dialogue.corpus.STEM+"-plan")+".json")
        report = RESULTS/(("4090-c161-name-final-s-joe" if joe else probe.dialogue.corpus.STEM+"-replay")+".json")
        for source, r in zip(read(plan)["records"], read(report)["records"], strict=True):
            if joe:
                assert source["c161"] == r["events"]["c161"] and source["index"] == r["index"]
                source = source["source"]
            pcm = probe.dialogue.piper.load_wav(source)
            assert base.poll.digest(pcm) == r["pcm_sha256"] == source["pcm_sha256"]
            meta = dict(label=r["label"],text=r["text"],variant=r["variant"]) if joe else dict(voice=r["voice"])
            yield r["index"], pcm, r["events"]["c161"], meta, source.get("timeline")


def verifier(raw, ends, student, matcher, config):
    codes = []
    for end in ends:
        patch = np.rint(prior.fit.room.window(raw,end).clip(0,666)*(255./666)).astype(np.uint8)
        codes.append(prior.outputs(student,patch))
    rows = {}
    for start in range(0,len(ends),96):
        xs = codes[start:start+96]
        for end, x, spelling in zip(ends[start:start+96],xs,matcher.scores(np.stack(xs)),strict=True):
            # A rejected spelling cannot pass the conjunctive gap rule.
            blank = None
            if spelling["score"] >= config["threshold"]:
                path = gap.diagnostic.trace(x,spelling["target"],matcher.vocab)
                assert path["normalized"] == spelling["target_value"]
                blank = path["interword_blank_frames"]
            rows[end] = dict(**spelling,gap=blank,accepted=blank is not None and blank<=config["max_gap"],
                             output_sha256=base.poll.digest(x))
    return rows


def tally(events, spans, group, total):
    outside = set(events)
    for span in spans:
        start, stop = span["start"], span["end"] + (6400 if group in ("fresh","dialogue") else 0)
        hits = [t for t in events if start<=t<stop]
        outside.difference_update(hits)
        total[span["address"]+".trials"] += 1
        total[span["address"]+".hits"] += bool(hits)
    total["outside"] += len(outside)


def main(group):
    measured = RESULTS/(probe.STEM+".json")
    reference = [s for s in read(measured)["streams"] if s["group"] == group]
    old = {r["index"]:r for r in reference}
    assert len(old) == len(reference)
    student, matcher, native = prior.model(), prior.Matcher(), prior.fit.native_frontend()
    config = read(gap.FIT)
    front, hashes = {}, {}
    for mode in fit.MODES:
        stem = fit.STEM+"-"+mode
        path = RESULTS / (stem+"-v1.tflite")
        assert sha(path)==read(RESULTS/(stem+"-export-v1.json"))["sha256"]
        front[mode] = REF_KERNEL(model_path=str(path))
        front[mode].allocate_tensors()
        hashes[mode] = sha(path)
    scene = group in ("matched","acoustics")
    reader, identity = base.scoring.word.load_reader() if scene else (None,None)
    archives = read(RESULTS/(gap.STEM+"-comparison.json"))["records"][group] if scene else None
    records, counts = [], defaultdict(Counter)
    for idx, pcm, c161, meta, spans in sources(group):
        parent = old.pop(idx)
        assert base.poll.digest(pcm) == parent["pcm_sha256"] and c161 == parent["c161"]
        raw,bands,clock = base.native_inputs(native,pcm)
        proposals, hashes_per_route, ends = {},{},{}
        for mode, vm in front.items():
            codes = base.infer(vm,bands)
            times = base.scoring.word.times(base.scoring.word.decode(codes,clock))
            proposals[mode] = [t for t in times if 0 < t <= len(pcm)]
            hashes_per_route[mode] = base.poll.digest(codes)
            ends[mode] = [t+base.scoring.word.WAIT for t in proposals[mode] if t+base.scoring.word.WAIT<=len(pcm)]
        checks = verifier(raw,sorted(set().union(*ends.values())),student,matcher,config)
        routes = dict(c161=c161,parent=parent["host"]["capped"])
        for mode, windows in ends.items():
            routes[mode] = base.scoring.merged(c161,[t for t in windows if checks[t]["accepted"]])
            counts[mode]["proposals"] += len(proposals[mode])
        r = dict(index=idx,pcm_sha256=parent["pcm_sha256"],source=meta,samples=len(pcm),events=routes,
                 proposals=proposals,checks=checks,trace_sha256=hashes_per_route,clock_sha256=base.poll.digest(clock))
        if scene:
            witness = archives[idx]
            assert witness["index"] == idx and witness["pcm_sha256"] == r["pcm_sha256"]
            cache = {int(k):v for k,v in witness["captures"].items()}
            cached = set(cache)
            kind = meta["role"] if group=="matched" else meta["acoustic"]
            r["scene"] = {}
        for route, ts in routes.items():
            total = counts[route]
            total["events"] += len(ts)
            total["samples"] += len(pcm)
            if scene:
                result = outcome(ts,meta,pcm,reader,cache)
                r["scene"][route] = result
                total[kind+".trials"] += 1
                total[kind+".timely"] += result["timely"]
                total[kind+".complete"] += result.get("complete_suffix",False)
            elif spans is not None:
                tally(ts,spans,group,total)
            elif group == "joe":
                total[meta["label"]+".trials"] += 1
                total[meta["label"]+".hits"] += bool(ts)
        if scene:
            r["new_transcripts"] = {k:v for k,v in cache.items() if k not in cached}
        records.append(r)
    assert not old
    save(RESULTS/f"{fit.STEM}-{group}.json",dict(records=records,summary=counts,asr=identity,models=hashes,
         training_sources_sha256=sha(RESULTS/(fit.STEM+"-sources.json")),parent_routes_sha256=sha(measured),
         character_sha256=read(prior.fit.EXPORT)["models"]["student"]["sha256"],gap_sha256=sha(gap.FIT),
         script_sha256=sha(Path(__file__)),policy="Full original continuous INT8 front-end clocks, five-frame "
         "sum threshold 1175, 360 ms wait, identical character model and gap gate. Preserve C161 notifications "
         "and shared cooldown. Compare at audio availability; new verifier windows have no measured board latency."))
    print(group,len(records),dict(counts["early"]),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("groups",nargs="+",choices=[*gap.study.GROUPS,"joe","dialogue"])
    for group in p.parse_args().groups:
        main(group)
