"""Test the frozen synthetic-speech mask against untouched C161 and realistic controls."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

import check_room_rnnoise as base
import music_proposal_replay as everyday
import train_speech_mask as fit

ch, gap, room = base.ch, base.gap, base.room
OUT = ch.RESULTS / (fit.STEM + "-replay.json")
INTERFERENCE = ch.RESULTS / "4090-c161-calibrated-echo-timing-interference.json"


def trace(native, pcm):
    codes, clock, _, _ = room.rooms.run_stream(native, pcm)
    events = room.rooms.times(room.rooms.decode(np.asarray(codes[0], dtype="u1"), np.asarray(clock)))
    return events, ch.digest(np.asarray(codes[0],dtype="u1"))


def causality(model, pcm):
    # A unity mask is the exact one-hop-delayed identity, up to PCM rounding.
    class Unity:
        def __call__(self, z, state):
            return torch.ones_like(z.real), state
    delayed = np.r_[np.zeros(fit.HOP,np.int16),pcm][:len(pcm)]
    np.testing.assert_array_equal(fit.enhance(Unity(),pcm), delayed)
    length = 160*125
    full = fit.enhance(model,pcm)
    short = fit.enhance(model,pcm[:length])
    changed = pcm.copy()
    changed[length:] = np.iinfo(np.int16).max
    other = fit.enhance(model,changed)
    diff = lambda a,b:int(np.abs(a.astype(int)-b.astype(int)).max())
    results = dict(prefix_samples=length, changed_future_max_lsb=diff(full[:length],other[:length]),
                   truncate_max_lsb=diff(full[:length],short), single_frame_max_lsb=diff(short,fit.enhance(model,pcm[:length],1)))
    assert results["changed_future_max_lsb"] == 0 and max(results[k] for k in ("truncate_max_lsb","single_frame_max_lsb")) <= 1
    return results


def main():
    model = fit.load_model()
    native = room.rooms.c161_binding()
    old = ch.read(base.OUT)
    assert ch.sha(room.rooms.MODEL) == old["model_sha256"]
    controls = ch.read(INTERFERENCE)
    fw = gap.uplink.firmware()
    assert fw == old["firmware"]
    reader, asr = ch.load_reader()
    assert asr == old["asr"]
    decoder = gap.uplink.decoder_api()
    records, interference, captures = [], [], []
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    pack = OUT.with_suffix(".zip")
    tmp = pack.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for i, ((s, x), reference) in enumerate(zip(base.inputs(),old["records"],strict=True)):
            assert s == reference["source"] and ch.digest(x)==s["pcm_sha256"]
            if i == 0:
                causal = causality(model,x)
            y = fit.enhance(model,x)
            row = dict(source=s, routes={})
            for name, signal in (("raw",x),("mask",y)):
                result = room.measure(native,signal,s["start"],s["end"])
                if name == "raw":
                    assert result["events"] == reference["routes"]["raw"]["events"]
                    assert result["codes_sha256"] == reference["routes"]["raw"]["codes_sha256"]
                clocks = [base.read_block(t if name=="raw" else (t+159)//160*160) for t in result["events"]]
                hits = [t for t in clocks if s["start"]<=t<len(x)]
                timely = bool(hits and hits[0] < s["end"])
                result.update(pcm_sha256=ch.digest(signal),delivered=clocks,hits=hits,timely=timely)
                tally = summary[s["kind"]][s["role"]][name]
                tally.update(trials=1, hits=bool(hits), timely=timely, pre_cue=sum(t<s["start"] for t in clocks))
                if hits and s["role"]=="lewis_nyx":
                    if name=="raw":
                        complete = reference["routes"]["raw"]["complete"]
                    else:
                        j = len(captures)
                        cap = gap.replay.capture(reader,decoder,work,logs,j,"mask",signal,hits[0],s["command"])
                        captures.append(dict(id=j,case=i,**cap))
                        result["capture"] = j
                        complete = cap["lanes"]["selected"]["asr"]["complete_suffix"]
                    tally.update(complete=complete,timely_complete=complete and timely)
                    result["complete"] = complete
                row["routes"][name] = result
            records.append(row)
        for group in ("daily","bot"):
            for (meta,x,prior,_,_), saved in zip(everyday.sources(group),controls["records"][group],strict=True):
                assert ch.digest(x) == saved["pcm_sha256"] and prior == saved["independent"]
                y = fit.enhance(model,x)
                row = dict(group=group,index=saved["index"],samples=len(x),source=meta["source"],routes={})
                for name,signal in (("raw",x),("mask",y)):
                    events, codes_hash = trace(native,signal)
                    if name=="raw":
                        assert events == prior
                    clocks = [base.read_block(t) for t in events]
                    spans = []
                    for span in meta["source"]["spans"]:
                        hits=[t for t in clocks if span["start"] <= t < span["end"]+6400]
                        spans.append(dict(text=span["text"],role=span["address"],hits=hits))
                        summary[group][span["address"]][name].update(trials=1,hits=bool(hits),events=len(hits))
                    row["routes"][name] = dict(events=events,delivered=clocks,spans=spans,
                                              pcm_sha256=ch.digest(signal),codes_sha256=codes_hash)
                interference.append(row)
        for p in sorted(work.glob("*-attempt*.gz")):
            logs.write(p,"retry/"+p.name)
    assert len(records)==64 and len(interference)==100
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for cap in captures:
            j=cap["id"]
            uart = z.read(f"opus/{j:03}-mask.log").decode().replace("\r\n","\n")
            for lane,name in enumerate(("selected","raw")):
                packets=gap.uplink.packets(uart,j,cap["emitted"]-cap["first"],lane)
                assert ch.digest(gap.uplink.decode(decoder,packets))==cap["lanes"][name]["decoded_sha256"]
    tmp.replace(pack)
    ch.save(OUT,dict(records=records,interference=interference,captures=captures,summary=summary,
        archive_sha256=ch.sha(pack), causal=causal,asr=asr,firmware=fw,model_sha256=old["model_sha256"],
        sources={p.name:ch.sha(p) for p in (fit.FIT,fit.WEIGHTS,INTERFERENCE,base.OUT,Path(__file__))},
        policy="Enhance from sample zero, retain one-hop algorithmic delay. Frozen C161 and same 256-sample "
        "read blocks. GPU inference, P4 simulated Opus. Raw captures reused on identical stream hashes. "
        "Daily/bot clocks attributed to existing spans through end+400ms. No continuous-background FAR claim."))


if __name__ == "__main__":
    main()
