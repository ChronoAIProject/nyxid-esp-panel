"""Locate the first command acoustically and transcribe the captured audio in six synthetic dialogues."""
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
import re

from pocketsphinx import Decoder

import replay_proposal_deadlines as study
from piper_command_onset_edits import DICTIONARY
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

dialogue = study.probe.dialogue


def command_onset(decoder, pcm, span):
    words = re.findall("[a-z]+",span["text"].lower())
    if words[0] == "hye":
        assert span["phones"].startswith("hˈaɪ ")
        words[0] = "hi"
    if words[1] == "nyx":
        assert "nˈɪks," in span["phones"]
        words[1] = "nicks"
    decoder.set_align_text(" ".join(words))
    clip = pcm[span["start"]:span["end"]].tobytes()
    for stage in ("words","phones"):
        if stage == "phones":
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(clip,False,True)
        decoder.end_utt()
    units = []
    for word in decoder.get_alignment():
        if word.name.startswith("<"):
            continue
        at = span["start"] + 160 * word.start
        units.append(dict(word=word.name,start=at,end=at+160*word.duration))
    assert [re.sub(r"\(\d+\)$","",w["word"]) for w in units] == words
    return units


def main():
    report = RESULTS/(study.fit.STEM+"-dialogue.json")
    original = RESULTS/(dialogue.corpus.STEM+"-replay.json")
    old_marks = RESULTS/(dialogue.corpus.STEM+"-onsets.json")
    witness = {(r["stream"],r["span"]):r for r in read(old_marks)["records"]}
    reader, identity = dialogue.piper.load_reader()
    assert identity == read(original)["asr"]
    counts, rows = defaultdict(Counter), []
    aligner_files = read(old_marks)["aligner_files"]
    assert all(sha(DICTIONARY/name)==digest for name,digest in aligner_files.items())
    for s, r, old in zip(read(dialogue.corpus.PLAN)["records"],read(report)["records"],read(original)["records"],strict=True):
        pcm = dialogue.piper.load_wav(s)
        assert study.base.poll.digest(pcm) == r["pcm_sha256"] == old["pcm_sha256"]
        routes = dict(**r["events"],official_willow=old["events"]["willow"])
        annotated = dialogue.piper.annotate(s,pcm,routes,[],[],reader)
        for j,(span,trial) in enumerate(zip(s["timeline"],annotated["rows"],strict=True)):
            if span["address"] not in ("Hey Nyx","Hye Nyx","Hey Nicks","Hey Willow"):
                continue
            decoder = Decoder(hmm=str(DICTIONARY/"en-us"),dict=str(DICTIONARY/"cmudict-en-us.dict"),
                              lm=None,cmn="batch",loglevel="ERROR")
            words = command_onset(decoder,pcm,span)
            command = words[2]["start"]
            if (r["index"],j) in witness:
                reference = witness[r["index"],j]["alignment"]
                assert command == next(w["start"] for w in reference if w["unit"]=="set")
            row = dict(stream=r["index"],span=j,voice=s["voice"],text=span["text"],address=span["address"],
                       pcm_sha256=trial["pcm_sha256"],words=words,command_at=command,routes={})
            for name, found in trial["routes"].items():
                willow = name=="official_willow"
                if willow != (span["address"]=="Hey Willow"):
                    continue
                stats = counts[name]
                stats["trials"] += 1
                hits = found["hits"]
                result = dict(events=hits)
                stats["hits"] += bool(hits)
                if hits:
                    begin = found["first"]
                    capture = annotated["transcripts"][found["asr"]]
                    delta = hits[0]-command
                    result.update(first_minus_command_samples=delta,capture_start=begin,complete_suffix=found["exact_command"],
                                  capture_text=capture["text"],capture_sha256=found["asr"])
                    stats["before_command"] += delta<0
                    stats["capture_contains_command_onset"] += begin<=command
                    stats["complete_suffix"] += found["exact_command"]
                row["routes"][name] = result
            rows.append(row)
    assert len(rows)==24
    save(RESULTS/(study.fit.STEM+"-dialogue-commands.json"),dict(records=rows,summary=counts,asr=identity,
         source_sha256=sha(report),original_sha256=sha(original),plan_sha256=sha(dialogue.corpus.PLAN),
         onset_witness_sha256=sha(old_marks),pocketsphinx=version("pocketsphinx"),
         aligner_files=aligner_files,script_sha256=sha(Path(__file__)),
         policy="Two English synthetic voices and three complete commands. Within each positive utterance, "
                "compare first notification to forced-aligned start of command word one (10 ms resolution). "
                "Unprompted transcription from two-second pre-roll through 400 ms past utterance end. "
                "Nyx clocks reflect audio availability; official Willow uses archived QEMU return clocks."))


if __name__ == "__main__":
    main()
