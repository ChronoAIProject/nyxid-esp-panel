"""Compress trained temporal lags onto the original clock without extra interpreter calls."""
import sys
from pathlib import Path

import flatbuffers
import numpy as np

from export_c161_dual_readout_v1 import graph
from fit_c161_history_gain_v1 import RESULTS, REF_KERNEL, read, save, sha
from check_c161_stream_stretch_v1 import materials, run_quantized
from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import frontend

STEM = "4090-c161-folded-time"


def fold(model, idx):
    g = model.subgraphs[0]
    op = g.operators[idx]
    source,w,b = (g.tensors[i] for i in op.inputs)
    previous = np.frombuffer(model.buffers[w.buffer].data,dtype=np.int8).reshape(w.shape)
    q = w.quantization
    assert np.all(q.zeroPoint==0) and np.all(b.quantization.zeroPoint==0)
    per_channel = len(q.scale)>1
    axes = [1]*previous.ndim
    axes[q.quantizedDimension]=len(q.scale)
    real = previous.astype(np.float64)*q.scale.reshape(axes)
    if real.ndim==2:
        real=real.reshape(real.shape[0],-1,60)
    count=real.shape[1]
    adjusted=np.zeros_like(real)
    for lag in range(count):
        # A virtual lag of one tick spans two-thirds of an original physical tick.
        n,rem=divmod(lag*2,3)
        adjusted[:,count-1-n] += (1-rem/3)*real[:,count-1-lag]
        if rem:
            adjusted[:,count-2-n] += rem/3*real[:,count-1-lag]
    np.testing.assert_allclose(adjusted.sum(1),real.sum(1),rtol=1e-10,atol=1e-10)
    adjusted=adjusted.reshape(previous.shape)
    summed_axes=tuple(i for i in range(previous.ndim) if not per_channel or i!=q.quantizedDimension)
    scale=np.atleast_1d(np.abs(adjusted).max(axis=summed_axes)/127).astype(np.float32)
    assert np.all(scale>0)
    new=np.rint(adjusted/scale.reshape(axes)).clip(-127,127).astype(np.int8)
    bias=np.frombuffer(model.buffers[b.buffer].data,dtype="<i4")*b.quantization.scale.astype(np.float64)
    sb=source.quantization.scale[0]*scale
    raw_bias=np.rint(bias/sb)
    assert np.max(np.abs(raw_bias))<=np.iinfo(np.int32).max
    model.buffers[w.buffer].data=np.frombuffer(new.tobytes(),np.uint8)
    model.buffers[b.buffer].data=np.frombuffer(raw_bias.astype("<i4").tobytes(),np.uint8)
    w.quantization.scale=scale
    b.quantization.scale=sb
    return dict(op=idx,time_steps=count,nonzero_before=int(np.count_nonzero(previous)),nonzero_after=int(np.count_nonzero(new)))


def build():
    spec=read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]
    original=RESULTS/spec["path"]
    assert sha(original)==spec["sha256"]
    variants=[]
    for name,ops in (("heads",(32,33)),("full",(17,19,22,25,28,32,33))):
        model=graph(original)
        log=[fold(model,i) for i in ops]
        builder=flatbuffers.Builder(85000)
        builder.Finish(model.Pack(builder),file_identifier=b"TFL3")
        path=RESULTS/f"{STEM}-{name}-v1.tflite"
        with path.open("xb") as f:
            f.write(bytes(builder.Output()))
        runtime=REF_KERNEL(model_path=str(path))
        runtime.allocate_tensors()
        assert len(runtime.get_input_details())==len(runtime.get_output_details())==1
        variants.append(dict(name=name,weights=log,path=path.name,bytes=path.stat().st_size,sha256=sha(path)))
    save(RESULTS/f"{STEM}-models-v1.json",dict(parent_sha256=sha(original),variants=variants))


def run(cohort):
    variants=read(RESULTS/f"{STEM}-models-v1.json")["variants"]
    arrays,rows={},[]
    for item,pcm,expected,_,_ in materials(cohort):
        values,clock=frontend(pcm)
        np.testing.assert_array_equal(clock,expected)
        events={}
        for v in variants:
            path=RESULTS/v["path"]
            assert sha(path)==v["sha256"]
            it=REF_KERNEL(model_path=str(path))
            scale,zero=it.get_input_details()[0]["quantization"]
            q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
            codes,calls=run_quantized(q,path,False)
            assert calls==len(clock)
            key=item.get("group","")+"_"+item["key"]+"_"+v["name"]
            arrays[key]=codes
            events[v["name"]]=times(decode(codes,clock))
        rows.append(dict(**item,events=events))
    out=RESULTS/f"{STEM}-{cohort}-scores-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-{cohort}-v1.json",dict(models={v["name"]:v["sha256"] for v in variants},scores_sha256=sha(out),records=rows))
    print(cohort,len(rows),flush=True)


if __name__=="__main__":
    for action in sys.argv[1:]:
        if action=="build":
            build()
        else:
            run(action)
