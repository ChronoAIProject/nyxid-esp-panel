"""Authenticate scene provenance, raw native returns, endpoints and all reported events."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from audit_c161_history_paths_v1 import arbitrate, decode
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_willow_matched_scenes_v1 import OUT, PLAN, SOURCE, STEM


def main():
    source, plan=read(SOURCE),read(PLAN)
    folder=Path(plan["records"][0]["path"]).parent
    assert source["records"]==plan["records"]
    for r in source["rows"]:
        stem=folder/f"{r['voice']}-{r['scene']}-{r['role']}"
        for ext,key in [(".wav","wav_sha256"),(".npy","native_sha256"),(".npz","alignment_sha256")]:
            assert sha(stem.with_suffix(ext))==r[key]
        native=np.load(stem.with_suffix(".npy"),allow_pickle=False)
        converted=resample_poly(native,160,441)
        pcm,rate=sf.read(stem.with_suffix(".wav"),dtype="int16")
        assert rate==16000
        np.testing.assert_array_equal(pcm,np.rint(np.clip(converted,-1,32767/32768)*32768).astype(np.int16))
    nyx=read(RESULTS/f"{STEM}-nyx-v1.json")
    assert nyx["plan_sha256"]==sha(PLAN)
    traces=RESULTS/f"{STEM}-nyx-scores-v1.npz"
    assert sha(traces)==nyx["scores_sha256"]
    dest=RESULTS/f"{STEM}-willow-evidence-v1.zip"
    detail,counts,files=[],defaultdict(Counter),{}
    with np.load(traces,allow_pickle=False) as z, ZipFile(dest,"x",compression=ZIP_DEFLATED) as pack:
        for i,(s,n) in enumerate(zip(plan["records"],nyx["records"],strict=True)):
            stage=OUT/f"{i:02d}"
            for name in ["report.json","serial.log","raw-returns.bin","stage-monitor.txt","host.log"]:
                path=stage/name
                member=f"{i:02d}/{name}"
                files[member]=sha(path)
                pack.write(path,member)
            w=read(stage/"report.json")
            assert w["stage"]==104 and w["execution_sha256"]==sha(PLAN)
            assert w["source_pcm_sha256"]==n["pcm_sha256"]==s["pcm_sha256"]
            assert w["actual_float32_bits"]==w["post_prime_getter_bits"]=="3f24dd2f"
            assert w["model"]=="wn9_heywillow_tts" and w["total_prime_samples"]==81600
            assert w["serial_sha256"]==files[f"{i:02d}/serial.log"] and w["raw_returns_sha256"]==files[f"{i:02d}/raw-returns.bin"]
            raw=np.fromfile(stage/"raw-returns.bin",dtype=np.uint8)
            assert len(raw)==w["input_calls"] and np.all(raw<=2)
            willow=np.minimum((np.flatnonzero(raw==2)+1)*512,w["input_samples"])-49344
            assert willow.tolist()==[e["sample_end"]-49344 for e in w["events"]]
            clock=z[f"{i}_clock"]
            nsteps={}
            for k in ["c161","ambient"]:
                nsteps[k]=decode(z[f"{i}_{k}"],clock)
                assert nsteps[k]==n["events"][k]
            assert arbitrate(nsteps["c161"],nsteps["ambient"])==(n["events"]["pair"],n["suppressed"])
            events={k:[e["relative_sample"] for e in v] for k,v in n["events"].items()}
            events["willow"]=willow.tolist()
            for k,ev in events.items():
                group=counts[f"{s['voice']}/{s['target_id']}/{k}"]
                group["scenes"]+=1
                group["target_segment_hits"]+=int(any(s["target_start"]<t<=s["samples"] for t in ev))
                group["chat_hits"]+=int(any(0<t<=s["chat_end"] for t in ev))
                group["name_hits"]+=int(any(s["name_start"]<t<=s["name_end"] for t in ev))
                group["gap_events"]+=sum(s["chat_end"]<t<=s["name_start"] or s["name_end"]<t<=s["target_start"] for t in ev)
                group["prime_events"]+=sum(t<=0 for t in ev)
            assert len(w["prime_returns"])==63 and all(v!=1 for _,v in w["prime_returns"])
            detail.append(dict(voice=s["voice"],scene=s["scene"],target=s["target_id"],command=s["carrier"],events=events,
                               target_start=s["target_start"],end=s["samples"],prefix_pcm_sha256=s["prefix_pcm_sha256"]))
    with ZipFile(dest) as pack:
        assert set(pack.namelist())==set(files) and len(files)==120
        for name,h in files.items():
            assert hashlib.sha256(pack.read(name)).hexdigest()==h
    save(RESULTS/f"{STEM}-review-v1.json",dict(plan_sha256=sha(PLAN),nyx_score_sha256=sha(traces),
         willow_archive_sha256=sha(dest),willow_members=files,counts=counts,records=detail))


if __name__=="__main__":
    main()
