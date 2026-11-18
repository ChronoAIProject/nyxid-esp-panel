"""Replay frozen C161 paths and official Willow on the same prospective scene PCM."""
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import arbitrate, decode
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from verify_c161_ambient_stream_v1 import BUILD, run_stream

ROOT = RESULTS.parent
STEM = "4090-c161-willow-matched-scenes"
SOURCE = RESULTS/f"{STEM}-sources-v1.json"
PLAN = RESULTS/f"{STEM}-run-plan-v1.json"
OUT = Path(".local/wakeword/c161-willow-matched-reference-v1")
sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
import run_willow_mac_matched_prime_v1 as official


def plan():
    parent = ROOT/"results/4090-willow-mac-matched-prime-plan-v1.json"
    old = read(parent)
    files = {str(p):sha(p) for p in [Path(__file__).resolve(),Path(official.__file__),official.SOURCE,SOURCE,parent]}
    rows = read(SOURCE)["records"]
    for path in old["runtime"].values():
        files[path] = sha(Path(path))
        assert files[path]==old["files"][path]
    for a,b in zip(rows[::2],rows[1::2],strict=True):
        assert (a["voice"],a["scene"],a["target_id"])==(b["voice"],b["scene"],"nyx") and b["target_id"]=="willow"
        assert a["prefix_pcm_sha256"]==b["prefix_pcm_sha256"] and a["target_start"]==b["target_start"]
        for s in [a,b]:
            path = Path(s["path"])
            pcm,rate=sf.read(path,dtype="int16")
            assert rate==16000 and sha(path)==s["sha256"] and len(pcm)==s["samples"]
            assert hashlib.sha256(pcm[:s["target_start"]].astype("<i2").tobytes()).hexdigest()==s["prefix_pcm_sha256"]
            files[str(path)]=s["sha256"]
    save(PLAN,dict(files=files,runtime=old["runtime"],records=rows,threshold_requests=["DEFAULT"]))


def nyx():
    live = read(RESULTS/"4090-c161-ambient-history-pair-live-v1.json")
    assert sha(MODEL)==MODEL_SHA and live["build"]["library_sha256"]==sha(BUILD/"stream.so")
    native=ctypes.CDLL(str((BUILD/"stream.so").resolve()))
    ptr=ctypes.c_void_p
    native.make_stream.argtypes=[]
    native.make_stream.restype=ptr
    native.stream_hop.argtypes=[ptr,ptr,ctypes.c_int,ptr]
    native.stream_events.argtypes=[ptr,ctypes.c_uint8,ctypes.c_uint8]
    native.destroy_stream.argtypes=[ptr]
    native.destroy_stream.restype=None
    traces,rows={},[]
    for i,r in enumerate(read(PLAN)["records"]):
        assert sha(Path(r["path"]))==r["sha256"]
        pcm,rate=sf.read(r["path"],dtype="int16")
        assert rate==16000 and hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest()==r["pcm_sha256"]
        codes,clock,emitted,suppressed=run_stream(native,pcm)
        clock=np.asarray(clock)
        traces[f"{i}_clock"]=clock
        events={}
        for label,v in zip(["c161","ambient"],codes,strict=True):
            traces[f"{i}_{label}"]=np.asarray(v,np.uint8)
            events[label]=decode(traces[f"{i}_{label}"],clock)
        assert arbitrate(events["c161"],events["ambient"])==(emitted,suppressed)
        rows.append(dict(index=i,pcm_sha256=r["pcm_sha256"],events=dict(**events,pair=emitted),suppressed=suppressed))
    path=RESULTS/f"{STEM}-nyx-scores-v1.npz"
    with path.open("xb") as f:np.savez_compressed(f,**traces)
    save(RESULTS/f"{STEM}-nyx-v1.json",dict(scores_sha256=sha(path),records=rows,plan_sha256=sha(PLAN)))


def willow():
    OUT.mkdir(exist_ok=True)
    p=read(PLAN)
    for f,h in p["files"].items():
        assert sha(Path(f))==h,f
    for i,r in enumerate(p["records"]):
        final=OUT/f"{i:02d}"
        if (final/"report.json").exists():
            assert read(final/"report.json")["execution_sha256"]==sha(PLAN)
            continue
        with tempfile.TemporaryDirectory(prefix="nyx-willow-") as t:
            work=Path(t)/"run"
            cmd=[sys.executable,__file__,"case","--model","willow","--threshold","DEFAULT",
                 "--record-index",str(i),"--output",str(work)]
            for key,path in p["runtime"].items():cmd += ["--"+key,path]
            try:
                subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            finally:
                if work.exists():shutil.move(str(work),str(final))
            assert read(final/"report.json")["source_pcm_sha256"]==r["pcm_sha256"]
            (final/"input.pcm").unlink()
        print(f"Willow {i+1}/{len(p['records'])}",flush=True)


def case():
    official.PLAN=PLAN
    official.__file__=str(Path(__file__).resolve())
    official.case()


if __name__=="__main__":
    action=sys.argv.pop(1)
    {"plan":plan,"nyx":nyx,"willow":willow,"case":case}[action]()
