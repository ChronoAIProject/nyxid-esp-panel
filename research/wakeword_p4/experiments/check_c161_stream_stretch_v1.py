"""Evaluate causal 3:2 input-feature expansion while retaining the original notification clock."""
import hashlib
import sys
from pathlib import Path

import numpy as np

from audit_c161_history_paths_v1 import decode, times
from check_c161_bounded_transfer_v1 import sources as transfer
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, read, save, sha
from replay_c161_vowel_name_pair_v1 import cases
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-stream-stretch"
MID = "4090-c161-midlevel"


def run_quantized(q, path, stretch):
    runtime = REF_KERNEL(model_path=str(path))
    runtime.allocate_tensors()
    inp, = runtime.get_input_details()
    out, = runtime.get_output_details()
    blocks = q.reshape(-1,3,40)
    physical = len(blocks)
    if stretch:
        # No future lookahead. Each first 10 ms frame is delivered once, each second one twice.
        count = np.arange(len(q),dtype=np.int64)%2+1
        available = count.cumsum()[2::3]//3
        virtual = np.repeat(q,count,axis=0)[:available[-1]*3].reshape(-1,3,40)
    else:
        virtual = blocks
        available = np.arange(1,physical+1)
    codes = np.empty(len(virtual),np.uint8)
    for i,frame in enumerate(virtual):
        runtime.set_tensor(inp["index"],frame[None])
        runtime.invoke()
        codes[i]=runtime.get_tensor(out["index"])[0,0]
    assert np.all(np.isin(np.diff(available,prepend=0),(1,2)))
    return codes[available-1],len(virtual)


def materials(name):
    if name=="transfer":
        report=read(RESULTS/"4090-c161-bounded-readout-transfer-v1.json")
        ref=RESULTS/"4090-c161-bounded-readout-transfer-scores-v1.npz"
        mid=RESULTS/f"{MID}-transfer-scores-v1.npz"
        assert sha(ref)==report["scores_sha256"] and sha(mid)==read(RESULTS/f"{MID}-transfer-v1.json")["scores_sha256"]
        with np.load(ref) as base,np.load(mid) as aux:
            for r,pcm in transfer():
                key=r["group"]+"_"+r["key"]
                yield r,pcm,base[key+"_clock"],base[key+"_c161"][0],aux[key][0]
    elif name=="forward":
        ref=RESULTS/"4090-c161-signed-forward-scores-v1.npz"
        mid=RESULTS/f"{MID}-forward-scores-v1.npz"
        prior=read(RESULTS/"4090-c161-signed-forward-v1.json")
        assert sha(ref)==prior["scores_sha256"] and sha(mid)==read(RESULTS/f"{MID}-forward-v1.json")["scores_sha256"]
        with np.load(ref) as base,np.load(mid) as aux:
            for i,r in enumerate(prior["records"]):
                item={k:r[k] for k in ("voice","scene","start","end","first_phone","pcm_sha256")}
                yield dict(key=str(i),role=r["target_id"],**item),load_wav(r),base[f"{i}_clock"],base[f"{i}_codes"][0],aux[str(i)][0]
    else:
        ref=RESULTS/f"{MID}-{name}-scores-v1.npz"
        prior=read(RESULTS/f"{MID}-{name}-v1.json")
        assert sha(ref)==prior["scores_sha256"]
        with np.load(ref) as z:
            for (item,pcm),r in zip(cases(name),prior["records"],strict=True):
                key=r["key"]
                assert item["key"]==key
                yield dict(**item,pcm_sha256=r["pcm_sha256"]),pcm,z[key+"_clock"],z[key+"_c161"][0],z[key+"_verifier"][0]


def main(name):
    fit=read(RESULTS/f"{MID}-training-v1.json")["reports"][0]
    paths={"c161":MODEL,"midlevel":RESULTS/fit["path"]}
    assert sha(MODEL)==MODEL_SHA and sha(paths["midlevel"])==fit["sha256"]
    scales={REF_KERNEL(model_path=str(path)).get_input_details()[0]["quantization"] for path in paths.values()}
    (scale,zero),=scales
    rows,arrays=[],{}
    for item,pcm,expected,c161,mid in materials(name):
        assert hashlib.sha256(pcm.tobytes()).hexdigest()==item["pcm_sha256"]
        values,clock=frontend(pcm)
        np.testing.assert_array_equal(clock,expected)
        q=np.clip(np.rint(values/scale)+zero,-128,127).astype(np.int8)
        key=item.get("group","")+"_"+item["key"]
        events={}
        for model,original in (("c161",c161),("midlevel",mid)):
            identity,calls=run_quantized(q,paths[model],False)
            np.testing.assert_array_equal(identity,original)
            stretched,total=run_quantized(q,paths[model],True)
            events[model]=dict(identity=times(decode(identity,clock)),stretch=times(decode(stretched,clock)))
            arrays[key+"_"+model]=stretched
        rows.append(dict(**item,physical_calls=calls,expanded_calls=total,events=events))
    out=RESULTS/f"{STEM}-{name}-scores-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-{name}-v1.json",dict(models={k:sha(p) for k,p in paths.items()},scores_sha256=sha(out),records=rows))
    print(name,len(rows),flush=True)


if __name__=="__main__":
    for cohort in sys.argv[1:]:
        main(cohort)
