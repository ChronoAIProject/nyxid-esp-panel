"""Keep a pre-speech noise estimate across completed WebRTC VAD speech intervals."""
import argparse
import ctypes
from importlib.metadata import version
from pathlib import Path
import subprocess

import micro_features_cpp
import numpy as np
import soundfile as sf
import webrtcvad

from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, RESULTS, features, read, save, score, sha
from review_c161_equal_length_history_v1 import PAIRED, SCORES, SOURCE
from run_parallel_context_streams_v1 import material

ROOT = Path("research/wakeword_p4")
STEM = "4090-c161-ambient-history"
CPP = ROOT / "frontend_probe/c161_ambient_history_v1.cpp"
UPSTREAM = Path(".local/wakeword/frontend-coefficients-source-v1/pymicro_features-2.0.2")
BUILD = Path(".local/wakeword/c161-ambient-history-build-v1")


def build():
    BUILD.mkdir(exist_ok=False)
    wheel = Path(micro_features_cpp.__file__).resolve()
    library = BUILD / "frontend.so"
    command = ["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-DFIXED_POINT=16",
               "-I", str(UPSTREAM), str(CPP), str(wheel), "-o", str(library)]
    subprocess.run(command, check=True)
    save(BUILD / "build.json", dict(command=command, wheel_sha256=sha(wheel), cpp_sha256=sha(CPP),
                                    headers={p.name:sha(p) for p in sorted(UPSTREAM.glob("tensorflow/lite/experimental/microfrontend/lib/*.h"))},
                                    library_sha256=sha(library), compiler=subprocess.check_output(["c++","--version"], text=True)))


def extract(native, pcm):
    full = np.concatenate((np.zeros(PRIME, np.int16), pcm))
    padded = np.pad(full, (0, -len(full) % 160))
    hops = padded.reshape(-1, 160)
    vad = webrtcvad.Vad(0)
    speech = np.fromiter((vad.is_speech(frame.tobytes(), 16000) for frame in hops), dtype=np.uint8, count=len(hops))
    starts = (np.flatnonzero(np.diff(speech.astype(int), prepend=0) == 1) + 1) * 160 - PRIME
    ends = (np.flatnonzero(np.diff(speech.astype(int), prepend=0) == -1) + 1) * 160 - PRIME
    raw = features(full)
    outputs = {}
    for treatment, name in [(0, "c161"), (1, "ambient")]:
        altered = np.empty_like(raw)
        result = native(padded.ctypes.data, speech.ctypes.data, len(hops), treatment, altered.ctypes.data, altered.size)
        assert result == 0
        if not treatment:
            np.testing.assert_array_equal(altered, raw)
        x, support, _ = prepare(altered, len(full))
        clock = np.minimum((support + 479) // 480 * 480, len(full)) - PRIME
        outputs[name] = score(x, clock)
    return outputs, clock, starts, ends


def run():
    library = BUILD / "frontend.so"
    evidence = read(BUILD / "build.json")
    assert sha(library) == evidence["library_sha256"] and sha(CPP) == evidence["cpp_sha256"]
    assert sha(MODEL) == MODEL_SHA and sha(SCORES) == read(PAIRED)["scores_sha256"]
    binding = ctypes.CDLL(str(library.resolve()))
    native = binding.ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    native.restype = ctypes.c_int
    arrays, streams, records = {}, [], []
    with np.load(SCORES, allow_pickle=False) as reference:
        def process(key, pcm):
            outputs, clock, starts, ends = extract(native, pcm)
            np.testing.assert_array_equal(outputs["c161"][0], reference[key + "_c161"])
            np.testing.assert_array_equal(clock, reference[key + "_clock"])
            arrays[key + "_scores"] = outputs["ambient"][0]
            arrays[key + "_speech_start_return"] = starts
            arrays[key + "_speech_end_return"] = ends
            return {arm:v[1] for arm,v in outputs.items()}

        source = read(SOURCE)
        for stream in source["streams"]:
            name = stream["name"]
            pcm, timeline = material(stream, source["gap_samples"])
            full = process(name + "_stream", pcm)
            streams.append(dict(name=name, events=full))
            items = {item["text_index"]:item for item in stream["items"]}
            for clip in timeline:
                if clip["kind"] != "clip":
                    continue
                index, start, end = clip["text_index"], clip["start"], clip["end"]
                item = items[index]
                target, rate = sf.read(item["wav"], dtype="int16")
                assert rate == 16000 and np.array_equal(target, pcm[start:end])
                isolated = process(f"{name}_{index}", target)
                continuous = {arm:[t for t in ev if start < t <= end] for arm,ev in full.items()}
                records.append(dict(stream=name, index=index, start=start, end=end, text=clip["text"],
                                    group=clip["group"], pcm_sha256=item["pcm_sha256"],
                                    isolated=isolated, continuous=continuous))
    assert len(records) == 144 and len(arrays) == 147 * 3
    archive = RESULTS / f"{STEM}-scores-v1.npz"
    with archive.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    save(RESULTS / f"{STEM}-v1.json", dict(build=evidence, vad_version=webrtcvad.__version__,
         frontend_version=version("pymicro-features"), scores_sha256=sha(archive), source_sha256=sha(SOURCE),
         control_scores_sha256=sha(SCORES), model_sha256=MODEL_SHA, streams=streams, records=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "run"])
    {"build":build, "run":run}[parser.parse_args().action]()
