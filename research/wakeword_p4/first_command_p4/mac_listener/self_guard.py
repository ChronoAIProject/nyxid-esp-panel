"""Verify both reference-gated notification routes, including late and early playback references."""
import argparse
import hashlib
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import replay as listener
import self_cue
from check_c161_v3_two_anchor_v1 import binding
from check_c161_v5_reference_guard import overlap_cases
from replay_c161_pcm_cues_v1 import detect
from transition import BUILD, cached_boot

STEM = self_cue.STEM+"-guard"
WORK = self_cue.WORK/"guard"


def streams():
    for i, mic, ref, meta in self_cue.trials():
        for setting, delta in (("self",0),("external",None)):
            actual = np.zeros_like(ref) if delta is None else ref
            yield f"{setting}-{i:03}",mic,actual,dict(meta,index=i,role=setting)
        if i in (0,24,76,80):
            for delta in (-256,256):
                shifted = np.roll(ref,delta)
                assert not np.any(shifted[:256]) and not np.any(shifted[-256:])
                yield f"skew-{i:03}-{delta}",mic,shifted,dict(meta,index=i,role="skew",skew_samples=delta)
    prior = listener.read(listener.RESULTS/(listener.STEM+".json"))["records"]
    for i,(src,old) in enumerate(zip(listener.read(listener.rooms.PLAN)["records"],prior,strict=True)):
        if src["role"]=="ordinary":
            mic = listener.load_wav(src)
            meta = dict(index=i,role="ordinary", baseline_c161=old["host_c161"],baseline_v5=old["host_v5"])
            yield f"ordinary-{i:03}",mic,np.zeros_like(mic),meta
            if old["host_c161"]:
                # Diagnostic electrical reference with no acoustic return. Observe deferral of real C161 hits.
                ref = np.rint(128*np.sin(2*np.pi*np.arange(len(mic))*440/16000)).astype("<i2")
                yield f"tone-{i:03}",mic,ref,dict(meta,role="reference_tone")
    for key,mic,ref,meta in overlap_cases():
        yield key,mic,ref,dict(meta,role="overlap")


def expected(events, refs, reference):
    active = np.flatnonzero(np.abs(reference.astype(np.int32))>=64)+1
    accepted = []
    for at in events:
        delayed = bool(np.any((at-8000<=active)&(active<=at)))
        end = at+480*delayed
        if not any(r<=end and r+3488>=at for r in refs):
            accepted.append(end)
    return accepted


def main(output):
    fw = BUILD/"nyx_mac_listener.bin"
    directory = WORK/listener.sha(fw)[:16]
    directory.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    archive = output/(STEM+"-events.zip")
    records = []
    previous = listener.RESULTS/(self_cue.STEM+".json")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        detector = binding(Path(tmp)/"probe.so",coded="v5")
        for i,(key,mic,ref,meta) in enumerate(streams()):
            payload = struct.pack("<3I",0x4752594e,i,len(mic))+mic.tobytes()+ref.tobytes()
            log = directory/(key+".log")
            cached_boot(BUILD,log,b"LISTENER PASS",payload)
            text = log.read_text()
            hits = [tuple(map(int,h.split())) for h in re.findall(r"^HITS (.+)$",text,re.M)]
            c161 = [t for _,t,_ in hits if t>=0]
            v5 = [t for _,_,t in hits if t>=0]
            if "baseline_c161" in meta:
                assert (c161,v5)==(meta["baseline_c161"],meta["baseline_v5"])
            refs = [e["t"] for e in detect(detector,[ref])]
            assert v5 == [e["t"] for e in detect(detector,[mic])]
            a,b = ([int(t) for t in re.findall("^"+label+r" (\d+)$",text,re.M)] for label in ("C161_ALLOW","V5_ALLOW"))
            assert a==expected(c161,refs,ref) and b==expected(v5,refs,ref)
            open_block = ((min(a+b)+255)//256)*256 if a or b else None
            actual = re.findall(r"^OPEN (\d+) (\d+) (\d+)$",text,re.M)
            assert [int(x[0]) for x in actual]==([open_block] if open_block else [])
            selected = "\n".join(l for l in text.splitlines() if l.startswith(("READY ","OPEN ","HITS ","CUE_ROUTE ","C161_ALLOW ","V5_ALLOW ","FINISH ","LISTENER PASS"))) + "\n"
            z.writestr(key+".log",selected)
            records.append(dict(key=key,**meta,mic_sha256=listener.digest(mic),reference_sha256=listener.digest(ref),
                reference_v5=refs,c161=c161,v5=v5,allowed_c161=a,allowed_v5=b,opened=open_block,
                first=int(actual[0][2]) if actual else None, uart_sha256=listener.sha(log),
                events_sha256=hashlib.sha256(selected.encode()).hexdigest()))
            if (i+1)%10==0:
                print(i+1,flush=True)
    archive.with_suffix(".tmp").replace(archive)
    listener.save(output/(STEM+".json"),dict(records=records,firmware_sha256=listener.sha(fw),
        header_sha256=listener.sha(listener.APP/"main/reference_wake.h"),previous_sha256=listener.sha(previous),
        events_sha256=listener.sha(archive), code_sha256=listener.sha(Path(__file__)),
        mac_commit=listener.MAC, model_sha256=listener.C161,
        policy="Either raw detector can open. During current or past 500 ms reference activity defer a positive for one 480-sample model stride. Cancel pending or current events throughout a recognized local v5 reference window. Silence passes immediately. Both final routes checked independently on the sample clock."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=WORK/"score")
    main(p.parse_args().out)
