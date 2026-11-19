"""Schedule the measured P4 lexical work in FIFO order and transcribe the resulting captures."""
import argparse
from collections import defaultdict

from check_c161_event_transcripts_v1 import choose
from check_c161_pocketsphinx_first_command import capture_commands, inputs, load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-lexical-p4"
INPUTS = RESULTS / f"{STEM}-inputs.json"
MAPPED = RESULTS / f"{STEM}-mapped.json"
SPEECH = RESULTS / "4090-c161-midlevel-pocketsphinx-v1.json"
BACKGROUND = RESULTS / "4090-c161-midlevel-pocketsphinx-background-v1.json"


def sample_to_us(sample):
    return sample * 1_000_000 // 16000


def us_to_sample(us):
    return (us * 16000 + 999_999) // 1_000_000


def service(src, case, worker_free_us):
    proposal = src["proposal"]
    arrival = sample_to_us(proposal)
    start = max(arrival, worker_free_us)
    clock = start + case["setup_us"]
    eligible = [m["returned"] for m in case["matches"] if proposal - 7200 <= m["end"] <= src["end"]]
    first_return = min(eligible, default=None)
    decision = None
    for i, cost in enumerate(case["block_cost_us"], 1):
        delivered = min(src["start"] + 480 * i, src["end"])
        clock = max(clock, sample_to_us(delivered)) + cost
        if delivered == first_return:
            decision = clock
    return dict(index=case["index"], proposal=proposal, queue_us=start-arrival, start_us=start,
                notify_us=decision, worker_free_us=clock+case["close_us"])


def schedule(input_path, measurement_path):
    manifest, measured = read(input_path), read(measurement_path)
    assert measured["input_sha256"] == sha(input_path)
    assert manifest["speech_report_sha256"] == sha(SPEECH)
    assert manifest["background_report_sha256"] == sha(BACKGROUND)
    windows = defaultdict(list)
    for src, case in zip(manifest["records"], measured["records"], strict=True):
        identity = src["cohort"], src["key"], src["source_pcm_sha256"]
        assert (case["cohort"], case["key"], case["proposal"]) == (*identity[:2], src["proposal"])
        assert case["matches_equal"] and case["confirmed"] == src["confirmed"]
        # Reproduce the independent firmware timing before introducing contention.
        first = service(src, case, 0)["notify_us"]
        delay = None if first is None else first - sample_to_us(src["proposal"])
        assert delay == case["modeled_confirm_us"]
        windows[identity].append((src, case))

    streams = {}
    for row in read(SPEECH)["records"] + read(BACKGROUND)["records"]:
        identity = row["cohort"], row["key"], row["pcm_sha256"]
        if identity[0] not in ("forward", "new", "melo", "primed", "jenny"):
            continue
        candidates = windows.pop(identity, [])
        assert [src["proposal"] for src, _ in candidates] == sorted(set(
            row["proposals"]["midlevel"] + row["proposals"]["expanded"]))
        jobs = []
        free_us = 0
        for src, case in candidates:
            job = service(src, case, free_us)
            jobs.append(job)
            free_us = job["worker_free_us"]
        confirms = [us_to_sample(j["notify_us"]) for j in jobs if j["notify_us"] is not None]
        streams[identity] = dict(cohort=identity[0], key=identity[1], pcm_sha256=identity[2],
                                 baseline=row["baseline"], jobs=jobs, events=choose(row["baseline"], confirms))
    assert not windows and sum(len(s["jobs"]) for s in streams.values()) == 62
    return streams


def main(compact):
    input_path = RESULTS / f"{STEM}-compact-inputs.json" if compact else INPUTS
    measurement_path = RESULTS / f"{STEM}-compact.json" if compact else MAPPED
    streams = schedule(input_path, measurement_path)
    by_audio = {(s["cohort"], s["pcm_sha256"]): s for s in streams.values()}
    model, files = load_reader()
    readings, targets = {}, []
    for cohort, src, span, _ in inputs(True):
        pcm = load_wav(src)
        assert digest(pcm) == src["pcm_sha256"]
        stream = by_audio[cohort, src["pcm_sha256"]]
        hits = [e for e in stream["events"] if span["start"] <= e["sample"] < span["end"]]
        captures = capture_commands(model, readings, pcm, span, hits[0]["sample"]) if hits else {}
        targets.append(dict(cohort=cohort, key=stream["key"], source_pcm_sha256=src["pcm_sha256"],
                            voice=src["voice"], span=span, hits=hits, captures=captures))
    assert len(targets) == 20
    result = dict(sources={p.name: sha(p) for p in (input_path, measurement_path, SPEECH, BACKGROUND)},
                  policy="One FIFO lexical worker completes each window including end_utt; original C161 runs independently. "
                         "Merge actual notifications with 67-step cooldown, capture on 160-sample boundary.",
                  asr="Whisper small.en CUDA FP16, unprompted; immediate or 32000 samples before capture boundary",
                  model_files=files, streams=list(streams.values()), targets=targets, transcripts=readings)
    prefix = f"{STEM}-compact" if compact else STEM
    save(RESULTS / f"{prefix}-fifo-first-command.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true")
    main(parser.parse_args().compact)
