"""Replay the frozen new word model and the same verifier over full speech, music and background clocks."""
import argparse
import ctypes
from pathlib import Path

import numpy as np
import soundfile as sf

import paired_proposal_denoise as study
import score_c161_paired_rooms as scoring
from check_c161_room_word_background import BUILD, CANDIDATE, infer
from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import PRIME, REF_KERNEL, RESULTS, read, save, sha
from train_c161_music_proposals import STEM

poll = scoring.poll


def sources(group):
    if group in ("acoustics", "music"):
        for meta, pcm, c161, old in study.materials(group):
            yield meta, pcm, c161, None, old
    elif group == "matched":
        report = read(scoring.MATCHED)
        assert report["plan_sha256"] == sha(scoring.MATCHED_PLAN)
        for s, r in zip(read(scoring.MATCHED_PLAN)["records"], report["records"], strict=True):
            pcm = poll.load_wav(s)
            assert poll.digest(pcm) == r["pcm_sha256"] == s["pcm_sha256"]
            yield dict(cue_at=s["start"], command_at=s["end"], expected=s["carrier"], voice=s["voice"],
                       role=s["target_id"], room=s["room"]), pcm, r["c161"], None, None
    elif group == "background":
        prior = read(RESULTS / f"{CANDIDATE}-background.json")
        for cohort in ("melo", "primed", "jenny"):
            for r in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
                if r["phase"] == "background":
                    s = r["source"]
                    key = f"{cohort}_{r['index']}"
                    expected = prior["records"][key]
                    pcm, rate = sf.read(s.get("wav_path") or s["path"], dtype="int16")
                    assert rate == 16000 and r["c161"] == [] and poll.digest(pcm) == s["pcm_sha256"] == expected["pcm_sha256"]
                    yield dict(key=key), pcm, [], expected["clock_sha256"], None
    else:
        for i, s, pcm, events in poll.live_sources(group):
            assert poll.digest(pcm) == s["pcm_sha256"]
            yield dict(index=i, source=s), pcm, events, None, None


def native_inputs(native, pcm):
    samples = PRIME+len(pcm)
    full = np.zeros(((samples+159)//160, 160), np.int16)
    full.ravel()[PRIME:samples] = pcm
    raw = np.empty((len(full)-2,40), np.uint16)
    quiet = np.zeros(len(full), np.uint8)
    assert native(full.ctypes.data, quiet.ctypes.data, len(full), 0, raw.ctypes.data, raw.size) == 0
    frames, support, _ = prepare(raw, samples)
    bands = np.rint(frames.reshape(-1,3,20,2).mean(axis=(1,3))*(255/26)).clip(0,255).astype(np.uint8)
    return raw, bands, np.minimum((support+479)//480*480, samples)-PRIME


def main(group):
    path = RESULTS / f"{STEM}-v1.tflite"
    assert sha(path) == read(RESULTS / f"{STEM}-export-v1.json")["sha256"]
    vm = REF_KERNEL(model_path=str(path))
    vm.allocate_tensors()
    lib = BUILD/"frontend.so"
    assert sha(lib) == read(BUILD/"build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    models, hashes = scoring.verifiers()
    reader, identity = scoring.word.load_reader() if group in ("acoustics","matched","fresh","background") else (None,None)
    rows = []
    for meta, pcm, c161, clock_hash, old in sources(group):
        raw, bands, clock = native_inputs(native, pcm)
        if clock_hash:
            assert poll.digest(clock) == clock_hash
        trace = infer(vm, bands)
        if not rows:
            clip = pcm[:50000]
            _, ref_inputs, ref_clock = native_inputs(native, clip)
            expected, expected_clock = scoring.word.score(vm, clip, vm.get_input_details()[0], vm.get_output_details()[0])
            np.testing.assert_array_equal(ref_clock, expected_clock)
            np.testing.assert_array_equal(infer(vm,ref_inputs), expected)
        proposals = scoring.word.times(scoring.word.decode(trace,clock))
        checks = []
        for t in proposals:
            end = t+scoring.word.WAIT
            if 0 < t and end <= len(pcm):
                window = np.rint(poll.fit.window(raw, end).clip(0,666)*(255./666)).astype(np.uint8)
                checks.append(dict(proposal=t,end=end,codes=models["teacher"].invoke(window).tolist()))
        events = scoring.merged(c161,[c["end"] for c in checks if c["codes"][0]>=0])
        r = dict(**meta, samples=len(pcm), pcm_sha256=poll.digest(pcm), c161=c161, proposals=proposals,
                 clock_sha256=poll.digest(clock), trace_sha256=poll.digest(trace), checks=checks, events=events)
        if group in ("acoustics","matched"):
            hit = bool(events and meta["cue_at"] <= events[0] < meta["command_at"])
            r["first_in_cue"] = hit
            if hit and r.get("role","ordinary") in ("ordinary","coded","willow"):
                begin = max(0, ((events[0]+255)//256*256)//160*160-32000)
                previous = old["captures"].get(str(begin)) if old else None
                r.update(begin=begin, capture=previous or study.diagnostic.scene.recognition(reader,pcm[begin:],meta["expected"]))
        elif group == "fresh":
            r.update(poll.annotate(meta["source"], pcm, dict(candidate=events), [], [], reader))
            del r["source"]["timeline"]
        elif group == "background":
            r["false_notifications"] = [dict(sample=t, begin=max(0,t-80000), end=min(len(pcm),t+48000),
                asr=study.diagnostic.scene.transcripts(reader,pcm[max(0,t-80000):t+48000])) for t in events]
        rows.append(r)
        if group == "background" and len(rows)%10 == 0:
            print(len(rows),flush=True)
    save(RESULTS / f"{STEM}-{group}.json", dict(records=rows, word_sha256=sha(path), verifier_sha256=hashes["teacher"],
         native_sha256=sha(lib), c161_sha256=scoring.MODEL_SHA, script_sha256=sha(Path(__file__)), asr=identity))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("matched","acoustics","music","background","bot","context","daily","fresh"))
    main(parser.parse_args().group)
