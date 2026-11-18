"""Two live C161 interpreters and VAD driven by the same 160-sample input hops."""
import ctypes
from pathlib import Path
import subprocess

import micro_features_cpp
import numpy as np
from pymicro_features import MicroFrontend
import webrtcvad

from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, PRIME, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import ROOT, STEM, UPSTREAM
from review_c161_equal_length_history_v1 import PAIRED, SCORES, SOURCE
from run_parallel_context_streams_v1 import material

BUILD = Path(".local/wakeword/c161-ambient-pair-live-v1")
CPP = ROOT / "frontend_probe/c161_ambient_stream_v1.cpp"


def binding():
    BUILD.mkdir(exist_ok=False)
    lib = BUILD / "stream.so"
    command = ["c++", "-std=c++17", "-O2", "-DFIXED_POINT=16", "-shared", "-fPIC", "-I", str(UPSTREAM),
               str(CPP), str(Path(micro_features_cpp.__file__).resolve()), "-o", str(lib)]
    subprocess.run(command, check=True)
    native = ctypes.CDLL(str(lib.resolve()))
    ptr = ctypes.c_void_p
    native.make_stream.argtypes = []
    native.make_stream.restype = ptr
    native.stream_hop.argtypes = [ptr,ptr,ctypes.c_int,ptr]
    native.stream_events.argtypes = [ptr,ctypes.c_uint8,ctypes.c_uint8]
    native.destroy_stream.argtypes = [ptr]
    native.destroy_stream.restype = None
    return native, dict(command=command, library_sha256=sha(lib), wrapper_sha256=sha(CPP),
                        snapshot_header_sha256=sha(ROOT/"portable/c161_ambient_history_v1.h"),
                        arbiter_sha256=sha(ROOT/"portable/c161_binary40_events_v1.h"))


def run_stream(native, pcm):
    frontend, vad = MicroFrontend(), webrtcvad.Vad(0)
    alternate = native.make_stream()
    if not alternate:
        raise MemoryError("alternate MicroFrontend allocation failed")
    try:
        models = []
        for _ in range(2):
            it = REF_KERNEL(model_path=str(MODEL))
            it.allocate_tensors()
            inp, = it.get_input_details()
            out, = it.get_output_details()
            models.append((it,inp,out))
        pending = [[],[]]
        codes = [[],[]]
        clock, emitted, suppressed = [], [], []
        temp = np.empty(40, np.uint16)
        def invoke(at):
            clock.append(at)
            for n,(it,inp,out) in enumerate(models):
                chunk = np.asarray(pending[n],dtype=np.float32)
                x = np.clip(chunk / 25.6,0,26)
                scale,zero = inp["quantization"]
                it.set_tensor(inp["index"], np.clip(np.rint(x/scale)+zero,-128,127).astype(np.int8)[None])
                it.invoke()
                codes[n].append(int(it.get_tensor(out["index"])[0,0]))
                pending[n].clear()
            flags = native.stream_events(alternate,codes[0][-1],codes[1][-1])
            for array,bits in [(emitted,(flags>>2)&3),(suppressed,(flags>>4)&3)]:
                for n,name in enumerate(["c161","ambient"]):
                    if bits & (1<<n):
                        array.append(dict(step=len(clock), relative_sample=at, model=name))
        full = np.r_[np.zeros(PRIME,np.int16), pcm]
        padded = np.pad(full,(0,-len(full)%160))
        for block_start in range(0,len(padded),480):
            at = min(block_start+480,len(full))-PRIME
            for frame in padded[block_start:block_start+480].reshape(-1,160):
                data = frame.tobytes()
                control = frontend.process_samples(data)
                received = native.stream_hop(alternate,frame.ctypes.data,vad.is_speech(data,16000),temp.ctypes.data)
                assert control.samples_read == 160 and received == len(control.features)
                if received:
                    pending[0].append(np.rint(np.asarray(control.features)*25.6).astype(np.uint16))
                    pending[1].append(temp.copy())
                    if len(pending[0]) == 3:
                        invoke(at)
        if pending[0]:
            for arm in pending:
                arm.extend([np.zeros(40,np.uint16)]*(3-len(arm)))
            invoke(len(pcm))
        return codes,clock,emitted,suppressed
    finally:
        native.destroy_stream(alternate)


def main():
    native, build = binding()
    assert sha(MODEL)==MODEL_SHA and sha(SCORES)==read(PAIRED)["scores_sha256"]
    alt_file = RESULTS/f"{STEM}-scores-v1.npz"
    assert sha(alt_file)==read(RESULTS/f"{STEM}-v1.json")["scores_sha256"]
    records = []
    with np.load(SCORES,allow_pickle=False) as base, np.load(alt_file,allow_pickle=False) as alt:
        def check(key,pcm):
            codes,clock,emitted,suppressed = run_stream(native,pcm)
            np.testing.assert_array_equal(codes[0],base[key+"_c161"])
            np.testing.assert_array_equal(codes[1],alt[key+"_scores"])
            np.testing.assert_array_equal(clock,base[key+"_clock"])
            records.append(dict(key=key,samples=len(pcm),calls=len(clock),emitted=emitted,suppressed=suppressed))
        source = read(SOURCE)
        for stream in source["streams"]:
            name = stream["name"]
            pcm,timeline = material(stream,source["gap_samples"])
            check(name+"_stream",pcm)
            for span in timeline:
                if span["kind"]=="clip":
                    check(f"{name}_{span['text_index']}",pcm[span["start"]:span["end"]])
    assert len(records)==147
    save(RESULTS/f"{STEM}-pair-live-v1.json",dict(build=build,model_sha256=MODEL_SHA,
         scores_sha256=[sha(SCORES),sha(alt_file)],calls=sum(r["calls"] for r in records),records=records))


if __name__=="__main__":
    main()
