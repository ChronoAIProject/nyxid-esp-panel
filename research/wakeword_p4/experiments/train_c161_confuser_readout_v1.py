"""Learn and export a nonnegative confuser penalty inside the one-output INT8 graph."""
import hashlib
from copy import deepcopy
from unittest.mock import patch

import flatbuffers
import numpy as np
import torch
from ai_edge_litert import schema_py_generated as schema
from torch.nn import functional as F

import train_c161_bounded_readout_v1 as bounded
from export_c161_dual_readout_v1 import graph
from fit_c161_history_gain_v1 import MODEL_SHA, REF_KERNEL, RESULTS, read, save, sha
from generate_c161_confuser_readout_v1 import SOURCE, STEM, WAVS

PARENT = bounded.PARENT
FINAL = RESULTS / f"{STEM}-single-v1.tflite"


def training_data():
    original = bounded.inputs

    def inputs():
        yield from original()
        source = read(SOURCE)
        assert sha(WAVS) == source["archive_sha256"]
        with np.load(WAVS) as archive:
            for r in source["records"]:
                pcm = archive[r["key"]]
                assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
                yield "new-" + r["key"], r["voice"], r["kind"] == "nyx", pcm

    with patch.object(bounded, "inputs", inputs):
        return bounded.collect()


def export(coefficients, offset):
    model = graph(PARENT)
    g = model.subgraphs[0]
    fc = g.operators[32]
    logistic = g.operators[33]
    assert list(fc.inputs) == [62, 14, 13] and list(logistic.inputs) == [63]
    assert list(fc.outputs) == [63] and list(g.outputs) == [70]
    concat = next(node for node in g.operators if model.operatorCodes[node.opcodeIndex].builtinCode ==
                  schema.BuiltinOperator.CONCATENATION)

    def tensor(source, name, data=None, shape=None, scale=None):
        t = deepcopy(g.tensors[source])
        t.name = name.encode()
        if shape is not None:
            t.shape = shape
        if scale is not None:
            t.quantization.scale = np.array([scale], np.float32)
        if data is not None:
            b = schema.BufferT()
            b.data = np.frombuffer(data.tobytes(), np.uint8)
            t.buffer = len(model.buffers)
            model.buffers.append(b)
        g.tensors.append(t)
        return len(g.tensors)-1

    penalty_weights = tensor(14, "confuser_weights", coefficients)
    penalty_bias = tensor(13, "confuser_bias", np.array([offset], "<i4"))
    penalty = tensor(63, "confuser_penalty")
    joined = tensor(63, "wake_and_penalty", shape=[1,2])
    combine_weights = tensor(14, "difference_weights", np.array([64,-64],np.int8), [1,2], 1/64)
    scale = g.tensors[63].quantization.scale[0]
    combine_bias = tensor(13, "difference_bias", np.array([0],"<i4"),scale=scale/64)
    verified = tensor(63, "verified_logit")
    first = deepcopy(fc)
    first.inputs = [62,penalty_weights,penalty_bias]
    first.outputs = [penalty]
    first.builtinOptions.fusedActivationFunction = schema.ActivationFunctionType.RELU
    join = deepcopy(concat)
    join.inputs = [63,penalty]
    join.outputs = [joined]
    join.builtinOptions.axis = 1
    last = deepcopy(fc)
    last.inputs = [joined,combine_weights,combine_bias]
    last.outputs = [verified]
    logistic.inputs = [verified]
    g.operators[33:33] = [first, join, last]
    builder = flatbuffers.Builder(65536)
    builder.Finish(model.Pack(builder),file_identifier=b"TFL3")
    with FINAL.open("xb") as f:
        f.write(bytes(builder.Output()))
    runtime = REF_KERNEL(model_path=str(FINAL))
    runtime.allocate_tensors()
    inp, = runtime.get_input_details()
    out, = runtime.get_output_details()
    assert inp["dtype"] == np.int8 and list(inp["shape"]) == [1,3,40]
    assert out["dtype"] == np.uint8 and out["quantization"] == (1/256,0) and list(out["shape"]) == [1,1]


def main():
    assert "4090" in torch.cuda.get_device_name()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    seed = 20261120
    rng = np.random.default_rng(seed)
    source = read(RESULTS/"4090-c161-paired-event-readout-training-v1.json")
    assert sha(PARENT) == next(r["sha256"] for r in source["reports"] if r["name"] == "speech")
    it = REF_KERNEL(model_path=str(PARENT), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    quant = {r["index"]: r["quantization"] for r in it.get_tensor_details()}
    sx, zx = quant[62]
    sw, zw = quant[14]
    sl, zl = quant[63]
    assert zx == -128 and zw == 0
    teacher = torch.tensor(it.get_tensor(14).ravel(), device="cuda", dtype=torch.float32) * sw
    bias = int(it.get_tensor(13)[0]) * sx * sw
    records, hidden = training_data()
    selections, windows = [], []
    for r, h in zip(records, hidden, strict=True):
        h_float = (h.float() - zx) * sx
        logits = ((h_float @ teacher+bias)/sl).round().clamp(-128-zl,127-zl)*sl
        peaks = logits.sigmoid().unfold(0,5,1).mean(-1)
        starts = []
        for _ in range(3):
            idx = int(peaks.argmax())
            starts.append(idx)
            peaks[max(0,idx-4):idx+5] = -1
        selections.append(dict(**r, starts=starts))
        windows.append(torch.stack([h[i:i+5] for i in starts]))
    del hidden
    x = (torch.stack(windows).float() - zx) * sx
    original = ((x @ teacher+bias)/sl).round().clamp(-128-zl,127-zl)*sl
    positive = np.array([r["positive"] for r in records])
    pos_ids = np.flatnonzero(positive)
    neg_ids = np.flatnonzero(~positive)
    w = torch.zeros_like(teacher,requires_grad=True)
    b = torch.tensor(1.,device="cuda",requires_grad=True)
    opt = torch.optim.Adam([w,b],lr=.0003)
    history = []
    for step in range(640):
        pi = rng.choice(pos_ids,48)
        ni = rng.choice(neg_ids,64)
        qw = (w/sw).round().clamp(-127,127)*sw
        eff = w+(qw-w).detach()
        def outputs(ids):
            # The intermediate ReLU uses the same INT8 scale and zero as the parent logit.
            penalty = (x[ids] @ eff+b).clamp(0,(127-zl)*sl)
            qp = (penalty/sl).round()*sl
            return original[ids]-(penalty+(qp-penalty).detach())
        pos = outputs(pi)[:,0,:]
        neg = outputs(ni)
        retention = (original[pi,0,:].clamp(max=4.)-pos).relu().square().mean()
        difficulty = F.softplus(neg-1.).mean(dim=-1).amax(dim=-1).topk(16).values.mean()
        loss = difficulty + 1.25*retention + .002*(eff/sw).square().mean()
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0,159,319,639):
            history.append(dict(step=step+1,loss=loss.item(),retention=retention.item(),negative=difficulty.item()))
    final = (w.detach()/sw).round().clamp(-127,127).cpu().numpy().astype(np.int8)
    offset = round(b.item()/(sx*sw))
    export(final,offset)
    ledger = RESULTS / f"{STEM}-windows-v1.json"
    save(ledger,dict(positive=len(pos_ids),negative=len(neg_ids),records=selections))
    save(RESULTS/f"{STEM}-training-v1.json",dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,
         baseline_sha256=MODEL_SHA,parent_sha256=sha(PARENT),source_sha256=sha(SOURCE),ledger_sha256=sha(ledger),
         seed=seed,steps=640,penalty_bias_int32=offset,nonzero=int(np.count_nonzero(final)),history=history,
         reports=[dict(name="verifier",path=FINAL.name,bytes=FINAL.stat().st_size,sha256=sha(FINAL))]))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["train"]:
        main()
    else:
        import replay_c161_vowel_name_pair_v1 as replay
        replay.STEM = STEM
        for cohort in sys.argv[1:]:
            replay.run(cohort)
