"""Swap the greeting and name between matched Joe period/exclamation renders."""
from pathlib import Path

import numpy as np
from pocketsphinx import Decoder

import replay_spelling_gap_dialogues as replay
from readout_name_parts import DICTIONARY
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = replay.base


def align(pcm, s):
    decoder = Decoder(hmm=str(DICTIONARY / "en-us"), dict=str(DICTIONARY / "cmudict-en-us.dict"),
                      lm=None, cmn="batch", loglevel="ERROR")
    decoder.set_align_text("hey nicks")
    a = s["lead_samples"]
    b = a+s["source_samples"]
    for stage in ("word","phone"):
        if stage == "phone":
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(pcm[a:b].tobytes(), False, True)
        decoder.end_utt()

    def mark(unit):
        t = a + 160*unit.start
        return dict(unit=unit.name, start=t, end=t+160*unit.duration)
    marks = [dict(**mark(w), phones=[mark(p) for p in w]) for w in decoder.get_alignment()]
    words = [w for w in marks if w["unit"] in ("hey", "nicks")]
    assert [w["unit"] for w in words] == ["hey","nicks"]
    return words[1]["start"], marks


def join(lead, name, lo, hi):
    # Same-source joins exactly reproduce the input, including length and clock.
    n = 64
    ramp = np.arange(n)/n
    shared = np.rint(lead[lo-n:lo]*(1-ramp) + name[hi-n:hi]*ramp).astype("<i2")
    return np.r_[lead[:lo-n], shared, name[hi:]]


def main():
    parent = RESULTS / "4090-c161-name-final-s-joe.json"
    inputs = RESULTS / "4090-c161-history-gain-joe-v1.json"
    ref = read(parent)
    assert ref["source_sha256"] == sha(inputs)
    pairs = {}
    for r in read(inputs)["records"]:
        s = r["source"]
        if s["text"] in ("Hey Nicks.", "Hey Nicks!") and s["variant"] == 0:
            assert s["snr_db"] is None and not s["synthetic_rir"]
            pairs.setdefault((s["length_scale"], s["draw"]), {})[s["text"][-1]] = r
    net = replay.Detector()
    model, proc, identity = replay.prior.fit.letters.teacher()
    records = []
    for (speed, draw), pair in sorted(pairs.items()):
        files, cuts, alignments = {}, {}, {}
        for k, r in pair.items():
            pcm = replay.piper.load_wav(r["source"])
            cuts[k], alignments[k] = align(pcm, r["source"])
            files[k] = pcm
        variants = {}
        for a in (".","!"):
            for b in (".","!"):
                x = join(files[a], files[b], cuts[a], cuts[b])
                events, checks = net.replay(x)
                raw, bands, clock = base.native_inputs(net.native, x)
                trace = base.infer(net.word, bands)
                sums = np.convolve(trace.astype(int), np.ones(5,dtype=int), "valid")
                best = int(sums.argmax())
                transcript = replay.prior.fit.reference.decode(model, proc, x)
                if a == b:
                    np.testing.assert_array_equal(x, files[a])
                    old = ref["records"][pair[a]["index"]]
                    assert events["gap_or_q8_with_five"] == old["events"]["gap_or_q8_with_five"]
                    assert checks == [{k:v for k,v in c.items() if k != "s_scores"} for c in old["checks"]]
                variants[a+b] = dict(pcm_sha256=base.poll.digest(x), events=events, checks=checks,
                                      front_max_five_sum=int(sums[best]), peak_end=int(clock[4:][best]), ctc=transcript)
        records.append(dict(speed=speed,draw=draw,source_ids={k:r["source"]["source_id"] for k,r in pair.items()},
                            alignments=alignments, name_start=cuts, variants=variants))
    assert len(records) == 6
    save(RESULTS/"4090-c161-joe-punctuation-parts.json", dict(records=records, source_sha256=sha(inputs),
         replay_sha256=sha(parent), script_sha256=sha(Path(__file__)), ctc=identity,
         policy="Six dry paired source draws, equal generation speed and draw index, independently aligned to "
         "hey nicks at 10 ms resolution. Keep the receiver greeting and pre-speech history, splice at N onset, "
         "then use donor name, tail and padding. Four ms linear crossfade immediately before the join. Both "
         "identity joins are sample-exact and must reproduce the previous live checks. All clock gates frozen."))


if __name__ == "__main__":
    main()
