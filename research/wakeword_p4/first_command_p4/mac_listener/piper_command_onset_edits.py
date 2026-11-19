"""Align the original Piper PCM and replay independent edits of the first /s/ of set."""
from pathlib import Path
from importlib.metadata import version
import re

import numpy as np
from pocketsphinx import Decoder
from readout_name_parts import DICTIONARY

import replay_spelling_gap_dialogues as replay
from check_c161_word_parts import mute
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

corpus = replay.corpus
digest = replay.base.poll.digest


def align(pcm, span):
    words = re.findall("[a-z]+", span["text"].lower())
    # Both archived Nyx spellings pronounce /nɪks/ in these sentences.
    if words[1] == "nyx":
        assert "nˈɪks," in span["phones"]
        words[1] = "nicks"
    decoder = Decoder(hmm=str(DICTIONARY / "en-us"), dict=str(DICTIONARY / "cmudict-en-us.dict"),
                      lm=None, cmn="batch", loglevel="ERROR")
    decoder.set_align_text(" ".join(words))
    clip = pcm[span["start"]:span["end"]].tobytes()
    for phones in (False, True):
        if phones:
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(clip, False, True)
        decoder.end_utt()

    def interval(unit):
        at = span["start"] + 160 * unit.start
        return dict(unit=unit.name, start=at, end=at + 160 * unit.duration)

    alignment = [dict(**interval(word), phones=[interval(phone) for phone in word])
                 for word in decoder.get_alignment()]
    spoken = [w for w in alignment if not w["unit"].startswith("<")]
    assert [re.sub(r"\(\d+\)$", "", w["unit"]) for w in spoken] == words
    assert words[2] == "set" and spoken[2]["phones"][0]["unit"] == "S"
    return alignment, spoken[2]["phones"][0]


def main():
    plan = read(corpus.PLAN)
    report_file = RESULTS / f"{corpus.STEM}-replay.json"
    report = read(report_file)
    assert report["plan_sha256"] == sha(corpus.PLAN) and sha(corpus.PACK) == plan["archive_sha256"]
    detector = replay.Detector()
    assert detector.identity == report["identity"]
    trials = []
    for i, src in enumerate(plan["records"]):
        if not src["carrier"].startswith("set "):
            continue
        original = replay.piper.load_wav(src)
        for j, s in enumerate(src["timeline"]):
            if s["address"] not in ("Hey Nick", "Hey Nyx", "Hey Nicks"):
                continue
            lo, stop = s["start"], s["end"] + 6400
            alignment, ph = align(original, s)
            a, b = ph["start"], ph["end"]
            delayed = b + 2400
            cuts = dict(original=None, command_s=(a, b), later_same_length=(delayed, delayed+b-a),
                        entire_command=(a, s["end"]))
            assert b > a and delayed+b-a <= s["end"]
            runs = {}
            # Start at stream origin for each edit. Stop at the next complete frontend and delivery tick.
            original_prefix = original[:((stop + 479) // 480) * 480]
            for name, cut in cuts.items():
                edited = mute(original_prefix, *cut) if cut else original_prefix
                if cut:
                    np.testing.assert_array_equal(original_prefix[:cut[0]], edited[:cut[0]])
                    np.testing.assert_array_equal(original_prefix[cut[1]:], edited[cut[1]:])
                routes, checks = detector.replay(edited)
                within = {k: [t for t in ts if lo <= t < stop] for k, ts in routes.items()}
                if name == "original":
                    old = report["records"][i]
                    assert all(ts == [t for t in old["events"][k] if lo <= t < stop] for k, ts in within.items())
                runs[name] = dict(cut=cut, pcm_sha256=digest(edited), events=within,
                                  checks=[c for c in checks if lo <= c.get("end", -1) < stop])
            trials.append(dict(stream=i, span=j, voice=src["voice"], text=s["text"], alignment=alignment, runs=runs))
    assert len(trials) == 6
    save(RESULTS / f"{corpus.STEM}-onsets.json", dict(records=trials, pocketsphinx=version("pocketsphinx"),
         aligner_files={str(p.relative_to(DICTIONARY)): sha(p) for p in
                       [DICTIONARY / "cmudict-en-us.dict", *(DICTIONARY / "en-us").iterdir()] if p.is_file()},
         plan_sha256=sha(corpus.PLAN), replay_sha256=sha(report_file), script_sha256=sha(Path(__file__)),
         policy="Full-utterance CMU word and phone forced alignment on archived PCM at 10 ms resolution. "
         "Independently mute the first /s/ of set, a same-length "
         "later interval, or the entire command from its first phone. 4 ms tapers, same continuous stream history."))


if __name__ == "__main__":
    main()
