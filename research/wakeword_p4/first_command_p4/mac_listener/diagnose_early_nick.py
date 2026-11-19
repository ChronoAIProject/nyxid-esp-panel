"""Localize daily Nick false wakes using word-aligned edits and frozen proposal clocks."""
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pocketsphinx import Decoder

import replay_early_proposal_readout as study
from check_c161_v5_code_gain import c161_binding, run_stream
from check_c161_word_parts import mute
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha
from readout_name_parts import DICTIONARY

replay, base = study.replay, study.base
STEM = "4090-c161-early-nick-parts"


def align(pcm, text):
    words = re.findall("[a-z]+", text.lower())
    decoder = Decoder(hmm=str(DICTIONARY / "en-us"), dict=str(DICTIONARY / "cmudict-en-us.dict"),
                      lm=None, cmn="batch", loglevel="ERROR")
    decoder.set_align_text(" ".join(words))
    for phones in (False, True):
        if phones:
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(pcm.tobytes(), False, True)
        decoder.end_utt()

    def bounds(unit):
        return dict(name=unit.name, start=160 * unit.start, end=160 * (unit.start + unit.duration))

    spoken = [dict(**bounds(w), phones=[bounds(p) for p in w]) for w in decoder.get_alignment()
              if not w.name.startswith("<")]
    assert [re.sub(r"\(\d+\)$", "", w["name"]) for w in spoken] == words
    return spoken


class Detector:
    def __init__(self):
        stem = replay.fit.STEM + "-early"
        front = RESULTS / (stem + "-v1.tflite")
        self.models, hashes = study.models()
        assert sha(MODEL) == MODEL_SHA and sha(front) == read(RESULTS / (stem + "-export-v1.json"))["sha256"]
        self.front = REF_KERNEL(model_path=str(front))
        self.front.allocate_tensors()
        self.native, self.c161 = replay.prior.fit.native_frontend(), c161_binding()
        self.student, self.matcher = replay.prior.model(), replay.prior.Matcher()
        self.gap = read(replay.gap.FIT)
        cutoff = read(RESULTS / (study.training.STEM + "-cutoff.json"))
        assert cutoff["models"] == hashes
        self.cutoff = cutoff["cutoff"]
        self.identity = dict(c161=MODEL_SHA, proposer=sha(front), five=hashes["trained"],
                             character=read(replay.prior.fit.EXPORT)["models"]["student"]["sha256"],
                             gap_sha256=sha(replay.gap.FIT), cutoff=self.cutoff)

    def score(self, pcm, anchors=()):
        word = base.scoring.word
        trace, clock, _, _ = run_stream(self.c161, pcm)
        c161 = word.times(word.decode(np.asarray(trace[0], np.uint8), np.asarray(clock)))
        raw, bands, clock = base.native_inputs(self.native, pcm)
        proposals = [t for t in word.times(word.decode(base.infer(self.front, bands), clock)) if 0 < t <= len(pcm)]
        live = [t + word.WAIT for t in proposals if t + word.WAIT <= len(pcm)]
        checks = replay.verifier(raw, sorted(set(live).union(anchors)), self.student, self.matcher, self.gap)
        for end, c in checks.items():
            patch = np.rint(replay.prior.fit.room.window(raw, end).clip(0, 666) * (255/666)).astype(np.uint8)
            c["five"] = self.models["trained"].invoke(patch).tolist()
            c["pass"] = c["accepted"] and c["five"][0] >= self.cutoff
        events = base.scoring.merged(c161, [t for t in live if checks[t]["pass"]])
        return dict(pcm_sha256=base.poll.digest(pcm), c161=c161, proposals=proposals, live_ends=live,
                    checks=checks, events=events)


def main():
    net = Detector()
    ref = RESULTS / (study.training.STEM + "-replay.json")
    old = read(ref)["records"]["daily"]
    rows = []
    for (meta, pcm, _, _, _), previous in zip(base.sources("daily"), old, strict=True):
        text = meta["source"]["spans"][0]["text"]
        if not text.startswith("Hey Nick,"):
            continue
        words = align(pcm, text)
        hey, nick, command = words[:3]
        assert [p["name"] for p in nick["phones"]] == ["N", "IH", "K"]
        original = net.score(pcm)
        assert original["pcm_sha256"] == previous["pcm_sha256"] and original["events"] == previous["events"]["calibrated"]
        for t, scores in previous["checks"].items():
            assert scores["trained"] == original["checks"][int(t)]["five"]
        edits = dict(greeting=(hey["start"], hey["end"]), name=(nick["start"], nick["end"]),
                     name_k=(nick["phones"][-1]["start"], nick["end"]), command=(command["start"], len(pcm)),
                     command_initial=(command["phones"][0]["start"], command["phones"][0]["end"]))
        runs = {}
        for label, (lo, hi) in edits.items():
            modified = mute(pcm, lo, hi)
            assert np.array_equal(modified[:lo], pcm[:lo]) and np.array_equal(modified[hi:], pcm[hi:])
            runs[label] = net.score(modified, original["live_ends"])
        rows.append(dict(index=previous["index"], text=text, alignment=words, edits=edits, original=original, runs=runs))
    assert len(rows) == 6
    save(RESULTS / (STEM + ".json"), dict(records=rows, models=net.identity, replay_sha256=sha(ref),
         script_sha256=sha(Path(__file__)), pocketsphinx=version("pocketsphinx"),
         aligner_files={str(p.relative_to(DICTIONARY)):sha(p) for p in [DICTIONARY / "cmudict-en-us.dict",
                        *(DICTIONARY / "en-us").iterdir()] if p.is_file()},
         policy="Use all six previously consumed Hey Nick daily recordings. Offline full-sentence word/phone "
                "alignment at 10 ms resolution. Mute each interval independently with the same four ms taper. "
                "Prime all detectors and reconstruct original proposals and both frozen branches; also read "
                "the original candidate clocks on edited PCM without using them to emit notifications."))


if __name__ == "__main__":
    main()
