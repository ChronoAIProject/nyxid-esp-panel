"""Share the auxiliary stream's C161 state and expose both real UINT8 scores by index."""
from copy import deepcopy
import ctypes
from pathlib import Path
from unittest.mock import patch

from ai_edge_litert import schema_py_generated as schema
import flatbuffers
import numpy as np

from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import cases
import replay_c161_ambient_history_v1 as frontend
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-dual-readout"
TARGET = RESULTS / f"{STEM}-v1.tflite"


def graph(path):
    return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))


def export():
    fit = read(RESULTS / "4090-c161-paired-event-readout-training-v1.json")
    selected, = [r for r in fit["reports"] if r["name"] == "speech"]
    alternate = RESULTS / selected["path"]
    assert sha(MODEL) == MODEL_SHA and sha(alternate) == selected["sha256"]
    model = graph(MODEL)
    aux = graph(alternate)
    g = model.subgraphs[0]
    weights = g.tensors[14]
    assert aux.subgraphs[0].tensors[14].buffer == weights.buffer
    assert all(a.data is None and b.data is None or np.array_equal(a.data, b.data)
               for i, (a,b) in enumerate(zip(model.buffers, aux.buffers, strict=True)) if i != weights.buffer)
    remap = {}
    for t in (14, 63, 64, 70):
        dup = deepcopy(g.tensors[t])
        dup.name = b"speech_" + dup.name
        if t == 14:
            dup.buffer = len(model.buffers)
            model.buffers.append(deepcopy(aux.buffers[weights.buffer]))
        remap[t] = len(g.tensors)
        g.tensors.append(dup)
    assert list(g.outputs) == [70]
    for index, inputs in ((32, [62,14,13]), (33, [63]), (44, [64])):
        node = deepcopy(g.operators[index])
        assert list(node.inputs) == inputs
        node.inputs = [remap.get(i, i) for i in inputs]
        node.outputs = [remap[i] for i in node.outputs]
        g.operators.append(node)
    # Hidden tensor 62 is still live at the second FC consumer in the graph.
    g.outputs = [70, remap[70]]
    sig, = model.signatureDefs
    output, = sig.outputs
    assert output.tensorIndex == 70
    second = deepcopy(output)
    second.name = b"speech_readout"
    second.tensorIndex = remap[70]
    sig.outputs.append(second)
    builder = flatbuffers.Builder(65536)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    raw = bytes(builder.Output())
    if TARGET.exists():
        assert TARGET.read_bytes() == raw
    else:
        with TARGET.open("xb") as f:
            f.write(raw)


def compare():
    lib = frontend.BUILD / "frontend.so"
    assert sha(lib) == read(frontend.BUILD / "build.json")["library_sha256"]
    native = ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    native.restype = ctypes.c_int
    # Use the original C161 frontend extraction and record the actual auxiliary input.
    def evaluate(pcm, reference_clock, expected):
        it = REF_KERNEL(model_path=str(TARGET))
        it.allocate_tensors()
        inp, = it.get_input_details()
        out = it.get_output_details()
        assert [r["index"] for r in out] == [70, 74]
        assert all(r["dtype"] == np.uint8 and r["quantization"] == (1/256, 0) for r in out)
        with patch.object(frontend, "score", lambda x, clock: x):
            features, clock, _, _ = frontend.extract(native, pcm)
        np.testing.assert_array_equal(clock, reference_clock)
        scale, zero = inp["quantization"]
        q = np.clip(np.rint(features["ambient"]/scale) + zero, -128, 127).astype(np.int8)
        for i, x in enumerate(q.reshape(-1,3,40)):
            it.set_tensor(inp["index"],x[None])
            it.invoke()
            codes = tuple(int(it.get_tensor(r["index"])[0,0]) for r in out)
            assert codes == (expected[0][i],expected[1][i]), (i,codes,expected[:,i])
        return len(clock)

    counts = {}
    for cohort in ("context", "willow", "daily", "mac", "joe", "generated"):
        stem = f"4090-c161-paired-event-readout-{cohort}"
        result = read(RESULTS / f"{stem}-v1.json")
        path = RESULTS / f"{stem}-scores-v1.npz"
        assert sha(path) == result["scores_sha256"]
        ticks = 0
        with np.load(path) as z:
            for (meta, pcm), r in zip(cases(cohort), result["records"], strict=True):
                k = meta["key"]
                assert k == r["key"]
                ticks += evaluate(pcm,z[k+"_clock"],np.stack([z[k+"_c161"][1],z[k+"_speech"][1]]))
        counts[cohort] = dict(streams=len(result["records"]), clocks=ticks)
        print(cohort, ticks, flush=True)
    stem = "4090-c161-four-voice-forward"
    report = read(RESULTS / f"{stem}-v1.json")
    review = read(RESULTS / f"{stem}-review-v1.json")
    base, other = RESULTS / f"{stem}-scores-v1.npz", RESULTS / f"{stem}-trained-scores-v1.npz"
    assert sha(base) == report["scores_sha256"] and sha(other) == review["scores_sha256"]
    ticks = 0
    with np.load(base) as b, np.load(other) as a:
        for i,r in enumerate(report["records"]):
            ticks += evaluate(load_wav(r), b[f"{i}_clock"], np.stack([b[f"{i}_codes"][1],a[str(i)][1]]))
    counts["forward"] = dict(streams=len(report["records"]),clocks=ticks)
    save(RESULTS / f"{STEM}-v1.json", dict(model_sha256=sha(TARGET), bytes=TARGET.stat().st_size,
         score_order=["original_ambient", "speech_ambient"], original_sha256=MODEL_SHA, cohorts=counts))


if __name__ == "__main__":
    export()
    compare()
