"""Locate the first failure in the ordinary-voice detector on the frozen acoustic ablations."""
from pathlib import Path

import numpy as np

import score_c161_paired_rooms as paired
import voice_cues as scene
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha


def outcome(events, checks, proposals, begin, end, mode, c161):
    timely = [c for c in checks if begin <= c["end"] < end]
    event = next(iter(events), None)
    if event is not None and event < begin:
        reason = "early_first_notification"
    elif event is not None and event < end:
        reason = "c161" if event in c161 else "verifier"
    elif not any(begin <= p < end for p in proposals) and not timely:
        reason = "no_proposal_in_cue"
    elif not timely:
        reason = "check_after_deadline"
    elif all(c["codes"][mode][0] < 0 for c in timely):
        reason = "verifier_rejected"
    else:
        raise AssertionError("An eligible confirmation was lost before the first notification")
    return dict(first_sample=event, outcome=reason)


def main():
    previous_path = RESULTS / (scene.STEM + "-rooms.json")
    previous = read(previous_path)
    assert previous["archive_sha256"] == sha(scene.PACK) and previous["model_sha256"] == scene.MODEL_SHA
    path = RESULTS / (paired.word.WORD + "-v1.tflite")
    assert sha(path) == read(RESULTS / (paired.word.WORD + "-export-v1.json"))["sha256"]
    vm = REF_KERNEL(model_path=str(path))
    vm.allocate_tensors()
    inp, = vm.get_input_details()
    out, = vm.get_output_details()
    models, hashes = paired.verifiers()
    reader, identity = scene.load_reader()
    rows = []
    for (s, pcm, _, _), old in zip(scene.scene_materials(), previous["records"], strict=True):
        if s["mode"] != "ordinary":
            continue
        assert scene.digest(pcm) == old["pcm_sha256"] and all(old[k] == v for k, v in s.items())
        begin, end = s["cue_at"], s["command_at"]
        trace, clock = paired.word.score(vm, pcm, inp, out)
        proposals = paired.word.times(paired.word.decode(trace, clock))
        peak5 = np.convolve(trace.astype(int), np.ones(5, int), mode="valid")
        peak = int(peak5[(clock[4:] >= begin) & (clock[4:] < end)].max())
        routes, checks = paired.evaluate(models, pcm, proposals, old["c161"])
        judgments, reads = {}, {}
        for name in models:
            result = outcome(routes[name], checks, proposals, begin, end, name, old["c161"])
            if result["outcome"] in ("c161", "verifier"):
                start = max(0, ((result["first_sample"]+255)//256*256)//160*160-32000)
                if start not in reads:
                    reads[start] = scene.recognition(reader, pcm[start:], s["expected"])
                result.update(capture_start=start, complete_suffix=reads[start]["complete_suffix"])
            judgments[name] = result
        rows.append(dict(**s, pcm_sha256=old["pcm_sha256"], c161=old["c161"], c161_peak5=old["c161_peak5"],
                         word_peak5=peak, proposals=proposals, checks=checks, routes=routes,
                         first=judgments, captures=reads))
    assert len(rows) == 64
    save(RESULTS / (paired.paired.STEM + "-acoustics.json"), dict(records=rows, prior_sha256=sha(previous_path),
         word_sha256=sha(path), model_sha256=hashes, asr=identity, script_sha256=sha(Path(__file__)),
         policy="Original command clocks and PCM for the 16 cases crossed with direct, noise only, room only and mixture. First notification must land before command. Split misses into absent proposal, check available too late, and negative INT8 Nyx decision."))


if __name__ == "__main__":
    main()
