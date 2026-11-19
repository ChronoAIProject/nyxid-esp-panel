"""Replay frozen word proposals, compare room verifiers, and preserve first-notification clocks."""
import argparse
from pathlib import Path

import check_c161_polling_proposals as word
import score_c161_polling_rooms as poll
import train_c161_paired_rooms as paired
from check_c161_address_grammar import merged
from check_c161_command_playback_level import recognition
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha

EXPORTS = RESULTS / (paired.STEM + "-exports.json")
MATCHED = RESULTS / "4090-c161-coded-voices-willow.json"
MATCHED_PLAN = RESULTS / "4090-c161-coded-voices-willow-plan.json"


def verifiers():
    exports = read(EXPORTS)
    assert exports["fit_sha256"] == sha(paired.FIT) and sha(MODEL) == MODEL_SHA
    prior = read(RESULTS / (poll.fit.STEM + "-exports.json"))["models"]["room"]
    models, hashes = {}, {}
    for mode, spec in dict(prior=prior, **exports["models"]).items():
        path = RESULTS / spec["file"]
        assert sha(path) == spec["sha256"]
        models[mode], hashes[mode] = poll.Model(path), spec["sha256"]
    return models, hashes


def evaluate(models, pcm, proposals, c161):
    ends = [(t, t+word.WAIT) for t in proposals if 0 < t and t+word.WAIT <= len(pcm)]
    raw = poll.fit.raw_features(pcm) if ends else None
    checks = []
    for t, end in ends:
        patch = poll.fit.window(raw, end)
        codes = {k:vm.invoke(patch).tolist() for k, vm in models.items()}
        checks.append(dict(proposal=t, end=end, codes=codes))
    routes = {k:merged(c161, [c["end"] for c in checks if c["codes"][k][0] >= 0]) for k in models}
    return dict(c161=c161, **routes), checks


def matched(models, reader):
    report, plan = read(MATCHED), read(MATCHED_PLAN)
    assert sha(MATCHED_PLAN) == report["plan_sha256"] and report["model_sha256"] == MODEL_SHA
    path = RESULTS / (word.WORD + "-v1.tflite")
    assert sha(path) == read(RESULTS / (word.WORD + "-export-v1.json"))["sha256"]
    vm = REF_KERNEL(model_path=str(path))
    vm.allocate_tensors()
    inp, = vm.get_input_details()
    out, = vm.get_output_details()
    rows = []
    for i, (s, old) in enumerate(zip(plan["records"], report["records"], strict=True)):
        pcm = poll.load_wav(s)
        assert poll.digest(pcm) == old["pcm_sha256"] == s["pcm_sha256"] and old["index"] == i
        codes, clock = word.score(vm, pcm, inp, out)
        proposals = word.times(word.decode(codes, clock))
        routes, checks = evaluate(models, pcm, proposals, old["c161"])
        capture, results = {}, {}
        for name, events in routes.items():
            hit = bool(events and s["start"] <= events[0] < s["end"])
            item = dict(events=events, first_in_cue=hit)
            if hit and s["target_id"] in ("ordinary", "coded", "willow"):
                begin = max(0, ((events[0]+255)//256*256)//160*160-32000)
                if begin not in capture:
                    capture[begin] = recognition(reader, pcm[begin:], s["carrier"])
                item.update(capture_start=begin, complete_suffix=capture[begin]["complete_suffix"])
            results[name] = item
        rows.append(dict(index=i, role=s["target_id"], room=s["room"], voice=s["voice"], carrier=s["carrier"],
                         pcm_sha256=s["pcm_sha256"], cue_at=s["start"], command_at=s["end"], proposals=proposals,
                         checks=checks, routes=results, captures=capture))
    return dict(records=rows, plan_sha256=sha(MATCHED_PLAN), reference_sha256=sha(MATCHED), word_sha256=sha(path))


def previous_group(group, models, reader):
    path = RESULTS / f"{word.STEM}-{group}.json"
    base = RESULTS / f"{poll.fit.STEM}-{group}.json"
    report = read(path)
    assert report["prior_sha256"] == sha(base) and report["wait_samples"] == word.WAIT
    assert report["word_sha256"] == sha(RESULTS / (word.WORD + "-v1.tflite"))
    original = read(base)["records"]
    if group == "fresh":
        plan = read(poll.fresh.PLAN)["records"]
        streams = ((i, s, poll.load_wav(s), r["routes"]["c161"]) for i,(s,r) in enumerate(zip(plan,original,strict=True)))
    else:
        streams = poll.live_sources(group)
    rows = []
    for (i, s, pcm, c161), old, ref in zip(streams, report["records"], original, strict=True):
        assert poll.digest(pcm) == old["pcm_sha256"] == ref["pcm_sha256"]
        assert i == old["index"] == ref["index"] and c161 == ref["routes"]["c161"]
        routes, checks = evaluate(models, pcm, old["proposals"], c161)
        assert routes["prior"] == old["routes"]["verified"]
        assert [c["codes"]["prior"] for c in checks] == [c["codes"] for c in old["checks"]]
        if group == "fresh":
            result = poll.annotate(s, pcm, routes, [], [], reader)
            for row in result["rows"]:
                del row["native_proposals"], row["expanded_proposals"]
        else:
            labels, outside = [], {k:set(ev) for k,ev in routes.items()}
            for span in old["rows"]:
                row = {k:v for k,v in span.items() if k != "routes"}
                row["routes"] = {}
                for k, events in routes.items():
                    hits = [t for t in events if row["start"] <= t < row["end"]]
                    outside[k].difference_update(hits)
                    row["routes"][k] = dict(hits=hits)
                labels.append(row)
            result = dict(rows=labels, outside={k:sorted(v) for k,v in outside.items()})
        rows.append(dict(index=i, meta=old["meta"], pcm_sha256=old["pcm_sha256"], proposals=old["proposals"],
                         checks=checks, routes=routes, **result))
    return dict(records=rows, prior_sha256=sha(path), base_sha256=sha(base), word_sha256=report["word_sha256"])


def main(group):
    models, hashes = verifiers()
    reader, asr = word.load_reader() if group in ("matched", "fresh") else (None, None)
    result = matched(models, reader) if group == "matched" else previous_group(group, models, reader)
    save(RESULTS / f"{paired.STEM}-{group}.json", dict(**result, model_sha256=hashes, c161_sha256=MODEL_SHA,
         export_sha256=sha(EXPORTS), asr=asr, script_sha256=sha(Path(__file__))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("matched", "fresh", "daily", "bot", "context"))
    main(parser.parse_args().group)
