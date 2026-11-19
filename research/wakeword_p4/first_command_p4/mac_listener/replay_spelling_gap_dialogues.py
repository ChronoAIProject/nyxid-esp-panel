"""Replay frozen C161, spelling, five-class, Q8 and official Willow on new continuous speech."""
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import proposal_lexical_readout as lexical
import score_c161_piper_spellings as piper
import score_spelling_gap as gap
import spelling_gap_dialogues as corpus
from check_c161_v5_code_gain import c161_binding, run_stream
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha

prior, base = gap.prior, gap.study.base


class Detector:
    def __init__(self):
        self.native = prior.fit.native_frontend()
        self.c161_frontend = c161_binding()
        parent = prior.fit.parent
        word = RESULTS / f"{parent.music.STEM}-v1.tflite"
        five = read(RESULTS / f"{lexical.positive.STEM}-exports.json")["models"]["trigger"]
        assert sha(word) == read(RESULTS / f"{parent.music.STEM}-export-v1.json")["sha256"]
        assert sha(RESULTS / five["file"]) == five["sha256"] and sha(MODEL) == MODEL_SHA
        self.word = REF_KERNEL(model_path=str(word))
        self.word.allocate_tensors()
        self.five = base.poll.Model(RESULTS / five["file"])
        self.char, self.matcher = prior.model(), prior.Matcher()
        self.decoder = piper.decoders()["name-contrast"]
        self.old, self.config = read(prior.CAL), read(gap.FIT)
        assert self.config["old_calibration_sha256"] == sha(prior.CAL)
        self.identity = dict(c161=MODEL_SHA, five=five["sha256"], word=sha(word),
                             character=read(prior.fit.EXPORT)["models"]["student"]["sha256"],
                             gap_fit=sha(gap.FIT), spelling_fit=sha(prior.CAL),
                             q8_fit=sha(RESULTS / "4090-c161-q8-name-contrast-fit.json"))

    def replay(self, pcm):
        trace, clock, _, _ = run_stream(self.c161_frontend, pcm)
        c161 = base.scoring.word.times(base.scoring.word.decode(np.asarray(trace[0], dtype=np.uint8), np.asarray(clock)))
        raw, bands, ticks = base.native_inputs(self.native, pcm)
        front = base.infer(self.word, bands)
        proposals = [t for t in base.scoring.word.times(base.scoring.word.decode(front, ticks)) if 0 < t <= len(pcm)]
        lexical_checks = lexical.confirmations(*self.decoder, pcm, proposals)
        accepted = {k: [] for k in ("paths", "gap", "five", "q8_and_five", "gap_and_five", "gap_or_q8_with_five")}
        checks = []
        for lc in lexical_checks:
            end = lc["proposal"] + base.scoring.word.WAIT
            if end > len(pcm):
                checks.append(dict(q8=lc))
                continue
            window = prior.fit.room.window(raw, end)
            x = np.rint(window.clip(0, 666) * (255. / 666)).astype(np.uint8)
            chars = prior.outputs(self.char, x)
            s, = self.matcher.scores(chars[None])
            path = gap.diagnostic.trace(chars, s["target"], self.matcher.vocab)
            assert path["normalized"] == s["target_value"]
            old = s["score"] >= self.old["threshold"]
            ok = s["score"] >= self.config["threshold"] and path["interword_blank_frames"] <= self.config["max_gap"]
            five_codes = self.five.invoke(x).tolist()
            if old:
                accepted["paths"].append(end)
            if ok:
                accepted["gap"].append(end)
            if five_codes[0] >= 0:
                accepted["five"].append(end)
                if ok:
                    accepted["gap_and_five"].append(end)
                if lc["confirmed"] is not None:
                    accepted["q8_and_five"].append(max(end, lc["confirmed"]))
            checks.append(dict(q8=lc, end=end, path=path, spelling=s, five=five_codes,
                               char_output_sha256=base.poll.digest(chars)))
        accepted["gap_or_q8_with_five"] = accepted["gap_and_five"] + accepted["q8_and_five"]
        return dict(c161=c161, **{k: base.scoring.merged(c161, ts) for k, ts in accepted.items()}), checks


def main():
    detector = Detector()
    # Exercise the live pipeline through an archived stream, including all gate clocks.
    old = read(RESULTS / f"{gap.STEM}-comparison.json")["records"]["context"][0]
    _, pcm, _, _, _ = next(base.sources("context"))
    assert base.poll.digest(pcm) == old["pcm_sha256"]
    check_routes, _ = detector.replay(pcm)
    assert all(check_routes[k] == times for k, times in old["events"].items() if k in check_routes)
    plan = read(corpus.PLAN)
    assert plan["archive_sha256"] == sha(corpus.PACK)
    reader, identity = piper.load_reader()
    summary, records = {}, []
    uart = RESULTS / f"{corpus.STEM}-willow.zip"
    with ZipFile(uart.with_suffix(".tmp"), "w", ZIP_DEFLATED) as archive:
        for i, src in enumerate(plan["records"]):
            pcm = piper.load_wav(src)
            routes, checks = detector.replay(pcm)
            with patch.multiple(piper, PLAN=corpus.PLAN, RUNS=corpus.RUNS):
                routes["willow"] = piper.official(i, src, archive)
            result = piper.annotate(src, pcm, routes, [], [], reader)
            for span in result["rows"]:
                for k, events in span["routes"].items():
                    total = summary.setdefault(k, Counter())
                    label = span["address"]
                    total[label + ".trials"] += 1
                    total[label + ".hit"] += bool(events["hits"])
                    total[label + ".complete_suffix"] += bool(events.get("exact_command"))
            for k, times in result["outside"].items():
                summary[k]["outside"] += len(times)
            records.append(dict(index=i, pcm_sha256=src["pcm_sha256"], voice=src["voice"], carrier=src["carrier"],
                                events=routes, checks=checks, **result))
            print(i, src["voice"], flush=True)
    uart.with_suffix(".tmp").replace(uart)
    save(RESULTS / f"{corpus.STEM}-replay.json", dict(records=records, summary=summary, plan_sha256=sha(corpus.PLAN),
         uart_sha256=sha(uart), identity=detector.identity, asr=identity, witness_sha256=old["pcm_sha256"],
         script_sha256=sha(Path(__file__)), policy="Original fits. Six uninterrupted dialogue streams, 2 s between "
         "phrases. An event from start to 400 ms after end counts once per utterance. Complete command requires "
         "a hit and an unprompted matching suffix from the two-second capture. Full-clip readings include misses. "
         "Notification times are audio availability, excluding Nyx verifier computation."))


if __name__ == "__main__":
    main()
