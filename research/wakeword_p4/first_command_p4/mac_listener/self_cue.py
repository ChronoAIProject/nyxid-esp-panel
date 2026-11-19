"""Attribute self-play wakes to each detector using matched mic waveforms and playback reference."""
import argparse
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import replay as listener
from check_c161_v3_two_anchor_v1 import binding
from replay_c161_pcm_cues_v1 import detect
from transition import BUILD, cached_boot

STEM = "4090-c161-mac-self-cue"
WORK = listener.WORK/"self-cue"


def trials():
    cue,rate = sf.read(listener.CUE,dtype="int16")
    assert rate == 16000
    baseline = listener.read(listener.RESULTS/(listener.STEM+".json"))
    assert baseline["files"][str(listener.rooms.PLAN)] == listener.sha(listener.rooms.PLAN)
    for i,(src,old) in enumerate(zip(listener.read(listener.rooms.PLAN)["records"],baseline["records"],strict=True)):
        if src["role"]!="v5":
            continue
        mic = listener.load_wav(src)
        assert listener.digest(cue) == src["source_cue_sha256"] and listener.digest(mic) == old["pcm_sha256"]
        ref = np.zeros_like(mic)
        ref[src["start"]:src["end"]] = np.rint(src["common_gain"]*cue)
        yield i, mic, ref, dict(snr_db=src["command_snr_db"],room=src["room"],voice=src["voice"],
            baseline_c161=old["host_c161"],baseline_v5=old["host_v5"],external_open=old["delivered"])


def main(output):
    fw = BUILD/"nyx_mac_listener.bin"
    directory = WORK/listener.sha(fw)[:16]
    directory.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    archive = output/(STEM+"-events.zip")
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        detector = binding(Path(tmp)/"probe.so",coded="v5")
        for i,mic,ref,meta in trials():
            log = directory/f"{i:03}.log"
            payload = struct.pack("<3I",0x5352594e,i,len(mic))+mic.tobytes()+ref.tobytes()
            cached_boot(BUILD,log,b"LISTENER PASS",payload)
            text = log.read_text()
            hits = [tuple(map(int,h.split())) for h in re.findall(r"^HITS (.+)$",text,re.M)]
            c161 = [t for _,t,_ in hits if t>=0]
            v5 = [t for _,_,t in hits if t>=0]
            assert c161 == meta["baseline_c161"] and v5 == meta["baseline_v5"]
            refs = [e["t"] for e in detect(detector,[ref])]
            routes = [tuple(map(int,h.split())) for h in re.findall(r"^CUE_ROUTE (.+)$",text,re.M)]
            assert routes == [(t,t if not any(0<=t-r<=3488 for r in refs) else -1) for t in v5]
            opened = re.findall(r"^OPEN (.+)$",text,re.M)
            assert len(opened) == int(bool(c161 or any(allowed>=0 for _,allowed in routes)))
            summary = "\n".join(l for l in text.splitlines() if l.startswith(("READY ","OPEN ","HITS ","CUE_ROUTE ","FINISH ","LISTENER PASS"))) + "\n"
            z.writestr(log.name,summary)
            rows.append(dict(index=i,**meta,reference_v5=refs,c161=c161,mic_v5=v5,routes=routes,
                self_open=int(opened[0].split()[0]) if opened else None, mic_sha256=listener.digest(mic),
                reference_sha256=listener.digest(ref),uart_sha256=listener.sha(log)))
    archive.with_suffix(".tmp").replace(archive)
    listener.save(output/(STEM+".json"), dict(records=rows,firmware_sha256=listener.sha(fw),mac_commit=listener.MAC,
        model_sha256=listener.C161,source_plan_sha256=listener.sha(listener.rooms.PLAN),
        mono_comparison_sha256=listener.sha(listener.RESULTS/(listener.STEM+".json")),events_sha256=listener.sha(archive),
        policy="Identical household mic input, externally spoken command and same background. Attribute the fixed cue alone to local playback in the synchronized right channel. Compare original external cue at zero reference. Archive unedited event lines; full local UART hashes recorded."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=WORK/"score")
    main(p.parse_args().out)
