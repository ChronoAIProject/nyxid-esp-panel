"""Replay 12 complete, distinct music tracks from the third pinned MUSAN music shard."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq
import requests
import soundfile as sf

from audit_c161_history_paths_v1 import decode, times
from collect_multispeaker_formal_v1 import predict
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, REF_KERNEL, RESULTS, frontend, read, save, sha

STEM="4090-c161-music-development"
ROOT=Path(".local/wakeword/c161-music-development-v1")
SHARD="music/train-00002-of-00010.parquet"
REV="76f9882cfa4475efe11508ac9aa32722f84ca5b7"
URL=f"https://huggingface.co/datasets/corypaik/musan/resolve/{REV}/{SHARD}"
SHARD_SHA="6e4273bbc0057691e40fad2cb79baf5f9a41bf00461d8c75e0c80faea91802d1"
SOURCE=RESULTS/f"{STEM}-sources-v1.json"


def acquire():
    ROOT.mkdir(exist_ok=True)
    index=Path(".local/wakeword/musan-music-features/ambient.json")
    prep=read(RESULTS/"4090-musan-music-preparation.json")
    assert sha(index)==prep["index_sha256"] and prep["all_training_only"]
    old=read(index)
    old_files={r["source_sha256"] for r in old}
    old_names={r["source_path"] for r in old}
    assert len(old_files)==132
    rows=[]
    assert shutil.disk_usage(ROOT).free > 101*(1<<30)
    with TemporaryDirectory(dir=ROOT) as temp:
        shard=Path(temp)/"music.parquet"
        with requests.get(URL,stream=True,timeout=(15,60)) as response, shard.open("wb") as f:
            response.raise_for_status()
            for chunk in response.iter_content(1<<20):
                f.write(chunk)
                assert f.tell()<=551196067
        assert shard.stat().st_size==551196067 and sha(shard)==SHARD_SHA
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1,use_threads=False):
            item,=batch.to_pylist()
            data=item["audio"]["bytes"]
            key=hashlib.sha256(data).hexdigest()
            assert item["label"]==0
            if key in old_files or item["path"] in old_names:
                continue
            pcm,rate=sf.read(io.BytesIO(data),dtype="int16")
            assert rate==16000 and pcm.ndim==1 and len(pcm)>0
            path=ROOT/f"{len(rows):02}.wav"
            if path.exists():
                assert sha(path)==key
            else:
                with path.open("xb") as f:
                    f.write(data)
            rows.append(dict(path=str(path),source_path=item["path"],source=item["source"],sha256=key,
                             pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),samples=len(pcm)))
            old_files.add(key)
            old_names.add(item["path"])
            if len(rows)==12:
                break
    assert len(rows)==len({r["pcm_sha256"] for r in rows})==12
    save(SOURCE,dict(revision=REV,url=URL,shard_sha256=SHARD_SHA,
         earlier_training_index_sha256=sha(index),records=rows))


def run():
    models={"c161":MODEL}
    for name in ("midlevel","rhythm-pair-native","rhythm-pair-rhythm"):
        fit=read(RESULTS/f"4090-c161-{name}-training-v1.json")["reports"][0]
        models[name]=RESULTS/fit["path"]
        assert sha(models[name])==fit["sha256"]
    assert sha(MODEL)==MODEL_SHA
    rows,arrays=[],{}
    for i,r in enumerate(read(SOURCE)["records"]):
        path=Path(r["path"])
        assert sha(path)==r["sha256"]
        pcm,rate=sf.read(path,dtype="int16")
        assert rate==16000 and len(pcm)==r["samples"] and hashlib.sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
        values,clock=frontend(pcm)
        events={}
        for name,model in models.items():
            _,codes,_=predict(values,str(model),REF_KERNEL)
            arrays[f"{i}_{name}"]=codes
            events[name]=times(decode(codes,clock))
        arrays[f"{i}_clock"]=clock
        rows.append(dict(source_path=r["source_path"],pcm_sha256=r["pcm_sha256"],samples=len(pcm),events=events))
        print(i,r["source_path"],{k:len(v) for k,v in events.items()},flush=True)
    out=RESULTS/f"{STEM}-scores-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-v1.json",dict(models={k:sha(p) for k,p in models.items()},source_sha256=sha(SOURCE),
         scores_sha256=sha(out),records=rows))


if __name__=="__main__":
    {"acquire":acquire,"run":run}[sys.argv[1]]()
