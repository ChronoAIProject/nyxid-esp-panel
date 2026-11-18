"""Replay frozen clean and hard-negative word classifiers over every sample of the 66 consumed recordings."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from collect_multispeaker_formal_v1 import prepare
from fit_c161_history_gain_v1 import PRIME, REF_KERNEL, RESULTS, read, save, sha
from replay_c161_ambient_history_v1 import BUILD
from train_c161_room_word import ORIGINAL
from train_c161_mined_room_word import PREFIX
from train_c161_word_window_v1 import bands

CANDIDATE = f"{PREFIX}-hard"
CACHE = Path(".local/wakeword/c161-room-word/background")


def native_features(native,pcm):
    samples = len(pcm)+PRIME
    data = np.zeros(((samples+159)//160,160),np.int16)
    data.ravel()[PRIME:samples] = pcm
    hops = len(data)
    raw = np.empty((hops-2,40),np.uint16)
    quiet = np.zeros(hops,np.uint8)
    assert native(data.ctypes.data,quiet.ctypes.data,hops,0,raw.ctypes.data,raw.size)==0
    del data, quiet
    values,support,_ = prepare(raw,samples)
    pooled = values.reshape(-1,3,20,2).mean(axis=(1,3))
    codes = np.rint(pooled*(255/26)).clip(0,255).astype(np.uint8)
    return codes,np.minimum((support+479)//480*480,samples)-PRIME


def infer(model,values,indices=None):
    inp,=model.get_input_details()
    out,=model.get_output_details()
    scale,zero=inp["quantization"]
    q=np.clip(np.rint(values.astype(np.float32)/255/scale)+zero,-128,127).astype(np.int8)
    q=np.pad(q,((31,0),(0,0)),constant_values=zero)
    if indices is None:
        indices = range(len(values))
    codes=np.empty(len(indices),np.uint8)
    for position, i in enumerate(indices):
        model.set_tensor(inp["index"],q[None,i:i+32,None,:])
        model.invoke()
        codes[position]=model.get_tensor(out["index"]).item()
    return codes


def main():
    CACHE.mkdir(exist_ok=True,parents=True)
    lib=BUILD/"frontend.so"
    assert sha(lib)==read(BUILD/"build.json")["library_sha256"]
    native=ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    models={}
    identities={}
    for label,stem in (("clean",ORIGINAL),("mined",CANDIDATE)):
        p=RESULTS/f"{stem}-v1.tflite"
        identities[label]=sha(p)
        assert identities[label]==read(RESULTS/f"{stem}-export-v1.json")["sha256"]
        models[label]=REF_KERNEL(model_path=str(p))
        models[label].allocate_tensors()
    raw_reports=[read(RESULTS/f"{s}-replay-v1.json") for s in (ORIGINAL,CANDIDATE)]
    rows, arrays={},{}
    for cohort in ("melo","primed","jenny"):
        p=RESULTS/f"4090-c161-history-gain-{cohort}-v1.json"
        for r in read(p)["records"]:
            if r["phase"]!="background":
                continue
            src=r["source"]
            path=Path(src.get("wav_path") or src["path"])
            key=f'{cohort}_{r["index"]}'
            pcm,rate=sf.read(path,dtype="int16")
            assert rate==16000 and pcm.shape==(src["samples"],) and hashlib.sha256(pcm.tobytes()).hexdigest()==src["pcm_sha256"]
            assert not r["c161"]
            cache=CACHE/f"{key}.npz"
            if not cache.exists():
                check=pcm[:60003]
                for got, expected in zip(native_features(native,check),bands(check),strict=True):
                    np.testing.assert_array_equal(got,expected)
                values,clock=native_features(native,pcm)
                traces={k:infer(model,values) for k,model in models.items()}
                tmp=cache.with_suffix(".tmp")
                with tmp.open("wb") as f:
                    np.savez_compressed(f,models=[*identities.values()],pcm_sha256=src["pcm_sha256"],clock=clock,**traces)
                tmp.replace(cache)
            with np.load(cache) as z:
                assert list(z["models"])==list(identities.values()) and z["pcm_sha256"].item()==src["pcm_sha256"]
                clock=z["clock"]
                clean,mined=z["clean"],z["mined"]
                hits={k:times(decode(trace,clock)) for k,trace in (("clean",clean),("mined",mined),("both",np.minimum(clean,mined)))}
                arrays[key+"_clean"],arrays[key+"_mined"]=clean,mined
            for label,report in zip(("clean","mined"),raw_reports,strict=True):
                for witness in report["records"]:
                    if witness["cohort"]=="transfer" and witness["meta"]["pcm_sha256"]==src["pcm_sha256"]:
                        assert witness["word"]==hits[label]
            rows[key]=dict(source=src.get("source_id",path.stem),cohort=cohort,pcm_sha256=src["pcm_sha256"],
                           samples=len(pcm),ticks=len(clock),clock_sha256=hashlib.sha256(clock.tobytes()).hexdigest(),events=hits)
            print(key,len(rows),[len(hits[k]) for k in ("clean","mined","both")],flush=True)
    assert len(rows)==66 and len({r["pcm_sha256"] for r in rows.values()})==66
    archive=RESULTS/f"{CANDIDATE}-background-scores.npz"
    with archive.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{CANDIDATE}-background.json",dict(model_sha256=identities,frontend_sha256=sha(lib),scores_sha256=sha(archive),
                samples=sum(r["samples"] for r in rows.values()),records=rows))


if __name__=="__main__":
    main()
