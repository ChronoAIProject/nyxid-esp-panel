"""Test the fixed 120 ms recording margin with the compact spelling model on Piper dialogue commands."""
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import check_c161_command_start as earlier
import proposal_dialogue_commands as study
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

replay = study.study
prior, base, gap = replay.prior, replay.base, replay.gap


def choose(pcm, hit, student, matcher, native, config):
    # Truncate the stream before extracting features; no post-notification audio
    # or command alignment can affect the proposed recording start.
    available = (hit+159)//160*160
    fallback = max(0,available-32000)
    raw = base.native_inputs(native,pcm[:available])[0]
    patch = np.rint(prior.fit.room.window(raw,available).clip(0,666)*(255./666)).astype(np.uint8)
    codes = prior.outputs(student,patch)
    spelling, = matcher.scores(codes[None])
    path = gap.diagnostic.trace(codes,spelling["target"],matcher.vocab)
    assert path["normalized"] == spelling["target_value"]
    allowed = spelling["score"] >= config["threshold"] and path["interword_blank_frames"] <= config["max_gap"]
    frame = path["characters"][-1]["last"]
    first = max(fallback,available-prior.fit.reference.WIDTH + 320*frame-earlier.MARGIN) if allowed else fallback
    return dict(first=first,fallback=fallback,accepted=allowed,available=available,last_frame=frame,
                spelling=spelling,gap=path["interword_blank_frames"],output_sha256=base.poll.digest(codes))


def main():
    report = RESULTS/(replay.fit.STEM+"-dialogue-commands.json")
    aligned = read(report)
    old = read(RESULTS/(study.dialogue.corpus.STEM+"-replay.json"))
    plan = read(study.dialogue.corpus.PLAN)
    reader, identity = base.scoring.word.load_reader()
    assert identity == aligned["asr"]
    student, matcher, native = prior.model(),prior.Matcher(),prior.fit.native_frontend()
    config = read(gap.FIT)
    counts, records = defaultdict(Counter),[]
    for row in aligned["records"]:
        sid, idx = row["stream"], row["span"]
        src = plan["records"][sid]
        pcm = study.dialogue.piper.load_wav(src)
        span = src["timeline"][idx]
        assert base.poll.digest(pcm[span["start"]:span["end"]]) == row["pcm_sha256"]
        stop = span["end"]+6400
        checks, readings = {},{}
        def transcribe(first):
            if first not in readings:
                clip = pcm[first:stop]
                readings[first] = dict(sha256=base.poll.digest(clip),**base.scoring.recognition(reader,clip,src["carrier"]))
            return readings[first]
        ref_start = max(0,row["command_at"]-earlier.MARGIN)
        oracle = transcribe(ref_start)
        full = old["records"][sid]["rows"][idx]["full_clip_asr"]
        assert full == row["pcm_sha256"]
        normalized = study.dialogue.piper.normalized
        want = normalized(src["carrier"])
        full_exact = normalized(old["records"][sid]["transcripts"][full]["text"])[-len(want):] == want
        group = "willow_reference" if row["address"]=="Hey Willow" else "nyx_reference"
        counts[group]["trials"] += 1
        counts[group]["full_utterance"] += full_exact
        counts[group]["aligned_margin"] += oracle["complete_suffix"]
        routes = {}
        for route, result in row["routes"].items():
            if route=="official_willow":
                continue
            total = counts[route]
            total["trials"] += 1
            if not result["events"]:
                continue
            hit = result["events"][0]
            if hit not in checks:
                checks[hit] = choose(pcm,hit,student,matcher,native,config)
            decision = checks[hit]
            assert decision["fallback"] == result["capture_start"]
            trimmed = transcribe(decision["first"])
            if decision["first"] == decision["fallback"]:
                assert trimmed["complete_suffix"] == result["complete_suffix"]
            total["hit"] += 1
            total["preroll_complete"] += result["complete_suffix"]
            total["trim_complete"] += trimmed["complete_suffix"]
            total["trimmed"] += decision["first"] != decision["fallback"]
            total["trim_after_command"] += decision["first"] > row["command_at"]
            total["oracle_complete_on_hits"] += oracle["complete_suffix"]
            routes[route] = dict(hit=hit,first=decision["first"],preroll=result["complete_suffix"],
                                 trimmed=trimmed["complete_suffix"])
        records.append(dict(stream=sid,span=idx,span_sha256=row["pcm_sha256"],command_at=row["command_at"],
                            oracle_start=ref_start,oracle_complete=oracle["complete_suffix"],full_complete=full_exact,
                            routes=routes,decisions=checks,captures=readings))
    save(RESULTS/(replay.fit.STEM+"-dialogue-trim.json"),dict(records=records,summary=counts,asr=identity,
         source_sha256=sha(report),margin_samples=earlier.MARGIN,gap_sha256=sha(gap.FIT),
         char_sha256=read(prior.fit.EXPORT)["models"]["student"]["sha256"],script_sha256=sha(Path(__file__)),
         policy="Evaluate a buffered read offset on existing first notifications. Infer the compact character "
                "outputs only from PCM delivered by that instant. If original spelling+gap accepts, choose the "
                "last frame of the winning name minus the pre-existing 120 ms margin; otherwise keep two seconds. "
                "Decoder frame step 320 samples, relative to the 1.44 s crop. Forced-aligned first command word "
                "minus 120 ms is a separate diagnostic using future speech. Full-utterance control uses original "
                "clip end; all other crops include the same 400 ms tail. Capture analysis, no new wake decisions."))


if __name__ == "__main__":
    main()
