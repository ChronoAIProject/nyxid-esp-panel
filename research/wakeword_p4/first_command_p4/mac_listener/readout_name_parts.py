"""Replay independent PCM edits of the common Max conflict and a nearby Nicks control."""
import ctypes
from importlib.metadata import version
from pathlib import Path
import re

import numpy as np
from pocketsphinx import Decoder, get_model_path

import music_proposal_replay as replay
import train_positive_readout as fit
from check_c161_word_parts import mute
from collect_multispeaker_formal_v1 import predict, prepare
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, REF_KERNEL, RESULTS, read, save, sha

CONTEXT = RESULTS / f"{fit.STEM}-context.json"
REPORT = RESULTS / "4090-c161-readout-name-parts.json"
DICTIONARY = Path(get_model_path()) / "en-us"
SENTENCES = {"Hey Max, can you carry the other bag upstairs?",
             "Hey Nicks, cancel the reminder about the laundry."}
runtime = replay.scoring.word
digest = replay.poll.digest


def align(pcm, span):
    decoder = Decoder(hmm=str(DICTIONARY / "en-us"), dict=str(DICTIONARY / "cmudict-en-us.dict"),
                      lm=None, cmn="batch", loglevel="ERROR")
    words = re.findall("[a-z]+", span["text"].lower())
    decoder.set_align_text(" ".join(words))
    clip = pcm[span["start"]:span["end"]].tobytes()
    for level in ("word", "phone"):
        if level == "phone":
            decoder.set_alignment()
        decoder.start_utt()
        decoder.process_raw(clip, False, True)
        decoder.end_utt()

    def bounds(unit):
        start = span["start"] + 160 * unit.start
        return dict(unit=unit.name, start=start, end=start + 160 * unit.duration)

    marks = [dict(**bounds(w), phones=[bounds(p) for p in w]) for w in decoder.get_alignment()]
    spoken = [m for m in marks if not m["unit"].startswith("<")]
    assert [re.sub(r"\(\d+\)$", "", m["unit"]) for m in spoken] == words
    hey, name, command = spoken[:3]
    expected = ["M", "AE", "K", "S"] if words[1] == "max" else ["N", "IH", "K", "S"]
    assert [p["unit"] for p in name["phones"]] == expected
    coda = name["phones"][2]["start"]
    return marks, dict(hey=(hey["start"], hey["end"]), name=(name["start"], name["end"]),
                       name_initial_and_vowel=(name["start"], coda), ks=(coda, name["end"]),
                       address=(span["start"], command["start"]), command=(command["start"], span["end"]))


def score(pcm, span, native, proposer, verifier, archived):
    # End on the next full delivery tick, so trimming the replay cannot add a partial tick inside the trial.
    stop = (span["end"] + 479) // 480 * 480
    pcm = pcm[:stop]
    raw, bands, clock = replay.native_inputs(native, pcm)
    values, _, _ = prepare(raw, PRIME + len(pcm))
    _, codes, _ = predict(values, str(MODEL), REF_KERNEL)
    c161 = runtime.times(runtime.decode(codes, clock))
    front = replay.infer(proposer, bands)
    proposals = runtime.times(runtime.decode(front, clock))

    def verify(t):
        end = t + runtime.WAIT
        patch = fit.parent.room.window(raw, end)
        x = np.rint(patch.clip(0, 666) * (255. / 666)).astype(np.uint8)
        return dict(proposal=t, end=end, codes=verifier.invoke(x).tolist(), patch_sha256=digest(x))

    checks = [verify(t) for t in proposals if 0 < t <= len(pcm) - runtime.WAIT]
    emitted = replay.scoring.merged(c161, [c["end"] for c in checks if c["codes"][0] >= 0])
    lo, hi = span["start"], span["end"]
    in_trial = lambda t: lo <= t < hi
    anchors = [verify(c["proposal"]) for c in archived["checks"] if in_trial(c["end"])]
    return dict(prefix_sha256=digest(pcm), through_sample=stop, proposals=[t for t in proposals if in_trial(t)],
                checks=[c for c in checks if in_trial(c["end"])], fixed_clock=anchors,
                c161=[t for t in c161 if in_trial(t)], events=[t for t in emitted if in_trial(t)])


def main():
    front = RESULTS / f"{fit.parent.music.STEM}-v1.tflite"
    front_hash = read(RESULTS / f"{fit.parent.music.STEM}-export-v1.json")["sha256"]
    spec = read(RESULTS / f"{fit.STEM}-exports.json")["models"]["trigger"]
    path = RESULTS / spec["file"]
    assert sha(front) == front_hash and sha(path) == spec["sha256"] and sha(MODEL) == MODEL_SHA
    lib = replay.BUILD / "frontend.so"
    assert sha(lib) == read(replay.BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    proposer = REF_KERNEL(model_path=str(front))
    proposer.allocate_tensors()
    verifier = replay.poll.Model(path)
    reader, reader_hashes = runtime.load_reader()
    rows = []
    for (meta, pcm, _, _, _), old in zip(replay.sources("context"), read(CONTEXT)["records"], strict=True):
        assert digest(pcm) == meta["source"]["pcm_sha256"] == old["pcm_sha256"]
        spans = [s for s in meta["source"]["spans"] if s["text"] in SENTENCES]
        assert len(spans) == 2 and {s["text"] for s in spans} == SENTENCES
        for span in spans:
            marks, edits = align(pcm, span)
            result = score(pcm, span, native, proposer, verifier, old)
            lo, hi = span["start"], span["end"]
            for name, previous in (("c161", old["c161"]), ("events", old["routes"]["trigger"]),
                                   ("proposals", old["proposals"])):
                assert result[name] == [t for t in previous if lo <= t < hi]
            checks = [(c["end"], c["codes"]["trigger"]) for c in old["checks"] if lo <= c["end"] < hi]
            assert [(c["end"], c["codes"]) for c in result["checks"]] == checks
            transcript = replay.study.diagnostic.scene.transcripts(reader, pcm[lo:hi])
            runs = dict(original=result)
            for name, (start, end) in edits.items():
                edited = mute(pcm, start, end)
                assert np.array_equal(edited[:start], pcm[:start]) and np.array_equal(edited[end:], pcm[end:])
                changed = score(edited, span, native, proposer, verifier, old)
                for k in ("c161", "proposals", "events"):
                    assert [t for t in changed[k] if t <= start] == [t for t in result[k] if t <= start]
                runs[name] = changed
            rows.append(dict(stream=meta["source"]["key"], stream_sha256=old["pcm_sha256"], span=span,
                             full_clip_asr=transcript, alignment=marks, edits=edits, runs=runs))
            print(meta["source"]["key"], span["text"], flush=True)
    save(REPORT, dict(records=rows, context_sha256=sha(CONTEXT), script_sha256=sha(Path(__file__)),
                      c161_sha256=MODEL_SHA, verifier_sha256=spec["sha256"], proposer_sha256=front_hash,
                      frontend_sha256=sha(lib), whisper_cuda=reader_hashes, pocketsphinx=version("pocketsphinx"),
                      aligner_files={str(p.relative_to(DICTIONARY)):sha(p) for p in
                                     [DICTIONARY / "cmudict-en-us.dict", *(DICTIONARY / "en-us").iterdir()] if p.is_file()},
                      policy="Each trial starts from the original unedited continuous stream and its usual prime. "
                      "Word and phone cuts from full-sentence forced alignment at 10 ms resolution, 4 ms tapers. "
                      "Rerun both INT8 paths through the utterance, including proposal, confirmation and merged "
                      "cooldown. Also rescore the original proposal windows for diagnosis even when an edit "
                      "removes the proposal. Developer-selected consumed examples; target is the nearest "
                      "preceding Nicks sentence in the same stream. ASR reads the original full clip."))


if __name__ == "__main__":
    main()
