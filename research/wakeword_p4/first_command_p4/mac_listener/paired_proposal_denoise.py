"""Apply causal SpeexDSP denoising only to the word proposer. Keep raw confirmation and capture."""
import argparse
import ctypes
from pathlib import Path

import numpy as np

import paired_noise_attribution as diagnostic
import score_c161_paired_rooms as scoring
from check_c161_speex_echo import LIB
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha

STEM = scoring.paired.STEM + "-speex"
MUSIC = RESULTS / "4090-c161-music-development-v1.json"
MUSIC_SOURCES = RESULTS / "4090-c161-music-development-sources-v1.json"
ACOUSTICS = RESULTS / (scoring.paired.STEM + "-acoustics.json")


def library():
    assert sha(LIB) == read(RESULTS / "4090-c161-speex-echo.json")["library_sha256"]
    api = ctypes.CDLL(str(LIB.resolve()))
    ptr, integer = ctypes.c_void_p, ctypes.c_int
    api.speex_preprocess_state_init.argtypes = [integer, integer]
    api.speex_preprocess_state_init.restype = ptr
    api.speex_preprocess_ctl.argtypes = [ptr, integer, ptr]
    api.speex_preprocess_run.argtypes = [ptr, ptr]
    api.speex_preprocess_state_destroy.argtypes = [ptr]
    api.speex_preprocess_state_destroy.restype = None
    return api


def denoise(api, pcm):
    # The streaming queue releases each 10 ms output at the end of its input block.
    # Keep the overlap-add delay. The final incomplete input block yields no extra tick.
    frame = 160
    state = api.speex_preprocess_state_init(frame, 16000)
    assert state
    try:
        for request, expected in ((1,1), (3,0), (19,-15)):
            value = ctypes.c_int()
            assert api.speex_preprocess_ctl(state, request, ctypes.byref(value)) == 0
            assert value.value == expected
        out = pcm[:len(pcm)//frame*frame].copy()
        for block in out.reshape(-1, frame):
            api.speex_preprocess_run(state, block.ctypes.data)
        return out
    finally:
        api.speex_preprocess_state_destroy(state)


def materials(group):
    if group == "acoustics":
        originals = iter(read(ACOUSTICS)["records"])
        for s, pcm, _, _ in diagnostic.scene.scene_materials():
            if s["mode"] != "ordinary":
                continue
            r = next(originals)
            assert scoring.poll.digest(pcm) == r["pcm_sha256"] and all(v == r[k] for k,v in s.items())
            yield s, pcm, r["c161"], r
        assert next(originals, None) is None
    else:
        previous = read(MUSIC)
        assert previous["source_sha256"] == sha(MUSIC_SOURCES)
        for s, r in zip(read(MUSIC_SOURCES)["records"], previous["records"], strict=True):
            pcm = scoring.poll.load_wav(s)
            assert scoring.poll.digest(pcm) == s["pcm_sha256"] == r["pcm_sha256"]
            yield dict(source=s["source_path"]), pcm, r["events"]["c161"], None


def main(group):
    path = RESULTS / (scoring.word.WORD + "-v1.tflite")
    assert sha(path) == read(RESULTS / (scoring.word.WORD + "-export-v1.json"))["sha256"]
    vm = REF_KERNEL(model_path=str(path))
    vm.allocate_tensors()
    inp, = vm.get_input_details()
    out, = vm.get_output_details()
    api = library()
    models, hashes = scoring.verifiers()
    reader, identity = scoring.word.load_reader() if group == "acoustics" else (None, None)
    rows = []
    for meta, pcm, c161, old in materials(group):
        row = dict(**meta, pcm_sha256=scoring.poll.digest(pcm), samples=len(pcm), c161=c161)
        filtered = denoise(api, pcm)
        row["denoised_sha256"] = scoring.poll.digest(filtered)
        for mode, audio in (("raw", pcm), ("denoised", filtered)):
            trace, clock = scoring.word.score(vm, audio, inp, out)
            proposals = scoring.word.times(scoring.word.decode(trace, clock))
            routes, checks = scoring.evaluate(models, pcm, proposals, c161)
            item = dict(proposals=proposals, checks=checks, routes=routes)
            if group == "acoustics":
                begin, end = meta["cue_at"], meta["command_at"]
                result = diagnostic.outcome(routes["teacher"], checks, proposals, begin, end, "teacher", c161)
                t = result["first_sample"]
                if result["outcome"] in ("verifier","c161"):
                    start = max(0, ((t+255)//256*256)//160*160-32000)
                    previous = old["captures"].get(str(start))
                    capture = previous or diagnostic.scene.recognition(reader, pcm[start:], meta["expected"])
                    result.update(capture_start=start, complete_suffix=capture["complete_suffix"])
                    item["capture"] = capture
                item["teacher_first"] = result
                if mode == "raw":
                    assert (proposals, routes, result) == (old["proposals"], old["routes"], old["first"]["teacher"])
            row[mode] = item
        rows.append(row)
    save(RESULTS / f"{STEM}-{group}.json", dict(records=rows, input_report_sha256=sha(ACOUSTICS if group == "acoustics" else MUSIC),
         script_sha256=sha(Path(__file__)), library_sha256=sha(LIB), word_sha256=sha(path), model_sha256=hashes,
         asr=identity, policy="Single-channel SpeexDSP 1.2.1 default denoise (-15 dB), 160-sample frames, AGC off. Cold start on each mic recording. Output is scored on its availability clock. The word proposer alone receives processed PCM. Original C161, verifier and captured command receive raw microphone PCM."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=("acoustics", "music"))
    main(parser.parse_args().group)
