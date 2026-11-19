"""Try a bounded speech-onset keyword search with both documented Piper first vowels."""
from collections import Counter
from pathlib import Path
import shutil
import sys

import numpy as np
from pocketsphinx import Decoder, Vad

from check_c161_event_transcripts_v1 import BOT, STRETCH, choose, event_inputs
from check_c161_piper_spellings import PLAN, STEM as PIPER
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_q8_keyword_map import FILES, LEXICAL
from train_c161_q8_name_contrast import ROOT as CONTRAST

sys.path.insert(0, str((RESULTS.parent / "pocketsphinx_p4_selftest").resolve()))
from compact import compact_definition, compact_weights

STEM = "4090-c161-q8-vad-pronunciations"
WORK = Path(".local/wakeword/c161-vad-pronunciations")
BLOCK = 480


def make_model():
    files = WORK / "files"
    am = files / "am"
    am.mkdir(parents=True, exist_ok=True)
    source = LEXICAL / "files/am"
    original = read(RESULTS / "4090-c161-lexical-p4-inputs.json")["files"]
    for name in ("mdef", "sendump"):
        assert sha(source / name) == original["am/" + name]
    dictionary = files / "three.dict"
    dictionary.write_text("hey HH EY\nhi HH AY\nnicks N IH K S\n")
    states, contexts = compact_definition(source / "mdef", am / "mdef", dictionary)
    compact_weights(source / "sendump", am / "sendump", states)
    old_states = read(RESULTS / "4090-c161-lexical-p4-compact-inputs.json")["original_state_ids"]
    fitted = CONTRAST / "am/sendump"
    assert sha(fitted) == read(RESULTS / "4090-c161-q8-name-contrast-fit.json")["adapted_sendump_sha256"]
    def matrix(data, columns):
        size = 3 * 128 * columns
        assert np.frombuffer(data[-size-8:-size], "<u4").tolist() == [128, columns]
        return np.frombuffer(data[-size:], np.uint8).reshape(3, 128, columns)
    data = (am / "sendump").read_bytes()
    weights = matrix(data, len(states)).copy()
    retained = [states.index(i) for i in old_states]
    assert np.array_equal(weights[:, :, retained], matrix((FILES / "am/sendump").read_bytes(), len(old_states)))
    weights[:, :, retained] = matrix(fitted.read_bytes(), len(old_states))
    (am / "sendump").write_bytes(data[:-weights.nbytes] + weights.tobytes())
    packing = read(RESULTS / "4090-c161-lexical-p4-q8-packing.json")
    for name in ("feat.params", "means", "variances", "noisedict", "transition_matrices"):
        assert sha(FILES / "am" / name) == packing["files"]["am/" + name]
        shutil.copyfile(FILES / "am" / name, am / name)
    phrases = files / "two.kws"
    phrases.write_text("hey nicks /1e-15/\nhi nicks /1e-15/\n")
    common = dict(hmm=str(am), dict=str(dictionary), lm=None, loglevel="ERROR", kws_threshold=1e-15)
    single = Decoder(**common, keyphrase="hey nicks")
    double = Decoder(**common, kws=str(phrases))
    assert single.lookup_word("hi") == double.lookup_word("hi") == "HH AY"
    return dict(hey=single, both=double), dict(states=states, contexts=contexts,
                files={str(p.relative_to(files)):sha(p) for p in files.rglob("*") if p.is_file()})


def windows(pcm):
    vad = Vad(mode=Vad.LOOSE, sample_rate=16000, frame_length=.03)
    assert vad.frame_bytes == 2 * BLOCK
    # One search immediately after each onset preceded by at least 450 ms of VAD silence.
    last_voiced = -16 * BLOCK
    for start in range(0, len(pcm), BLOCK):
        frame = pcm[start:start+BLOCK]
        if len(frame) < BLOCK:
            frame = np.pad(frame, (0, BLOCK-len(frame)))
        if vad.is_speech(frame.tobytes()):
            if start - last_voiced >= 16 * BLOCK:
                onset = min(start + BLOCK, len(pcm))
                yield onset, max(0, start - 10 * BLOCK), min(len(pcm), onset + 50 * BLOCK)
            last_voiced = start


def match(d, cmn, pcm, onset, start, end):
    d.reinit_feat()
    d.set_cmn(cmn)
    d.start_utt()
    seen, hits = set(), []
    for at in range(start, end, BLOCK):
        stop = min(at + BLOCK, end)
        d.process_raw(pcm[at:stop].tobytes(), False, False)
        for seg in d.seg() or ():
            word = seg.word.strip()
            mark = (word, seg.start_frame, seg.end_frame)
            if mark not in seen:
                assert word in ("hey nicks", "hi nicks")
                seen.add(mark)
                hits.append(dict(word=word, available=max(onset, stop), start=start+160*seg.start_frame,
                                 end=start+160*(seg.end_frame+1), prob=seg.prob))
    d.end_utt()
    return hits


def inputs():
    prior = read(RESULTS / f"{PIPER}-evaluation.json")
    for index, (src, previous) in enumerate(zip(read(PLAN)["records"], prior["records"], strict=True)):
        assert src["pcm_sha256"] == previous["pcm_sha256"]
        yield "piper", str(index), load_wav(src), previous["events"]["name-contrast_dual"], [
            dict(group=s["address"], text=s["text"], start=s["start"], end=s["end"]+6400) for s in src["timeline"]]
    base = {(r["cohort"],r["key"]):r for r in read(RESULTS / "4090-c161-q8-name-contrast-evaluation.json")["ordinary"]}
    for group in ("daily", "context", "bot"):
        for meta, key, _, pcm in event_inputs(group):
            prev = base[group, key]
            assert digest(pcm) == prev["pcm_sha256"]
            baseline = [e["sample"] for e in prev["events"]["dual"]]
            if group == "daily":
                spans = [dict(group="daily", text=meta["text"], start=0, end=len(pcm))]
            elif group == "bot":
                spans = meta["spans"]
            else:
                spans = [s for s in meta["timeline"] if s["kind"] == "clip"]
            yield group, key, pcm, baseline, spans


def main():
    decoders, model = make_model()
    cmn = {k:v.get_cmn() for k,v in decoders.items()}
    totals = Counter()
    records = []
    for group, key, pcm, baseline, spans in inputs():
        candidates = []
        seen = {name:[] for name in decoders}
        for onset, start, end in windows(pcm):
            record = dict(onset=onset, start=start, end=end, matches={})
            for name, d in decoders.items():
                found = match(d, cmn[name], pcm, onset, start, end)
                seen[name].extend(h["available"] for h in found)
                record["matches"][name] = found
            candidates.append(record)
        routes = {name:[e["sample"] for e in choose(baseline, hits)] for name, hits in seen.items()}
        routes["base"] = baseline
        outside = {name:set(values) for name,values in routes.items()}
        labels = []
        for s in spans:
            label = dict(group=s["group"], text=s["text"], start=s["start"], end=s["end"], hits={})
            for name, events in routes.items():
                hits = [t for t in events if s["start"] <= t < s["end"]]
                outside[name].difference_update(hits)
                label["hits"][name] = hits
                totals[group, s["group"], name] += bool(hits)
            totals[group, s["group"], "count"] += 1
            labels.append(label)
        records.append(dict(cohort=group, key=key, pcm_sha256=digest(pcm), samples=len(pcm), windows=candidates,
                            labels=labels, outside={name:sorted(x) for name,x in outside.items()}, events=routes))
    sources = [PLAN, RESULTS/f"{PIPER}-evaluation.json", RESULTS/"4090-c161-q8-name-contrast-evaluation.json",
               RESULTS/f"{BOT}-v1.json", *(RESULTS/f"{STRETCH}-{g}-v1.json" for g in ("context", "daily")),
               RESULTS/"4090-c161-lexical-p4-q8-packing.json", RESULTS/"4090-c161-lexical-p4-compact-inputs.json"]
    save(RESULTS/f"{STEM}.json", dict(sources={p.name:sha(p) for p in sources}, code_sha256=sha(Path(__file__)),
         policy="30ms WebRTC loose VAD; after 450ms silence look back 300ms, then run 1500ms. Merge with existing dual-speed name-contrast notifications at their audio clocks, 67-step cooldown; no P4 task scheduling assumed.",
         model=model, totals=[dict(cohort=g, group=label, route=r, value=v) for (g,label,r),v in sorted(totals.items())],
         records=records))


if __name__ == "__main__":
    main()
