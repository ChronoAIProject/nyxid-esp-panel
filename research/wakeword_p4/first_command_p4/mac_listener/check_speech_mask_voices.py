"""Run the frozen mask before C161 on all eight previously selected voices and their paired controls."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from scipy.signal import fftconvolve

import check_speech_mask as check
import replay_room_spoken_selection as prior

ch, gap, room, fit = check.ch, check.gap, check.room, check.fit
OUT = ch.RESULTS / (fit.STEM + "-voices.json")


def main():
    model, native = fit.load_model(), room.rooms.c161_binding()
    selected = ch.read(prior.selection.FIT)
    old = ch.read(prior.OUT)
    assert ch.sha(prior.selection.FIT) == old["sources"][prior.selection.FIT.name]
    assert ch.sha(prior.selection.PACK) == selected["pool_sha256"]
    bank = {r["index"]: pcm for r,pcm in prior.selection.pool()}
    reader, identity = ch.load_reader()
    assert identity == old["asr"]
    decoder = gap.uplink.decoder_api()
    fw = gap.uplink.firmware()
    assert fw == old["firmware"]
    records, captures = [], []
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    pack = OUT.with_suffix(".zip")
    tmp = pack.with_suffix(".tmp")
    with TemporaryDirectory() as folder, ZipFile(tmp,"w",ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for (s, _, kind, h, bg, cmd), previous in zip(room.room_inputs(), old["records"], strict=True):
            assert previous["silence_sha256"] == s["pcm_sha256"] and previous["command_at"] == s["end"]
            for voice, choices in selected["choices"].items():
                for role in ("target","nick"):
                    index = choices["selected"][role]
                    cue = bank[index]
                    end = s["end"]
                    start = end-len(cue)
                    wet = fftconvolve(cue.astype(float), h)
                    mix = bg + cmd
                    mix[start:start+len(wet)] += wet
                    x = room.quantize(mix * s["gain"])
                    source = previous["cues"][str(index)]
                    assert ch.digest(x) == source["pcm_sha256"]
                    raw = source["routes"]["c161"]
                    y = fit.enhance(model,x)
                    trace = room.measure(native,y,start,end)
                    clocks = [check.base.read_block((t+159)//160*160) for t in trace["events"]]
                    hits = [t for t in clocks if start<=t<len(y)]
                    masked = dict(**trace, delivered=clocks,hits=hits,pcm_sha256=ch.digest(y))
                    masked["timely"] = bool(hits and hits[0]<end)
                    if hits and role=="target":
                        i = len(captures)
                        cap = gap.replay.capture(reader,decoder,work,logs,i,"voice",y,hits[0],s["carrier"])
                        captures.append(dict(id=i,record=len(records),**cap))
                        masked.update(capture=i, complete=cap["lanes"]["selected"]["asr"]["complete_suffix"])
                    for name,r in (("raw",raw),("mask",masked)):
                        c = summary[kind][role][name]
                        c.update(trials=1,hits=bool(r["hits"]),timely=r.get("timely",False))
                        if role=="target":
                            c.update(complete=r.get("complete",False),timely_complete=r.get("complete",False) and r.get("timely",False))
                    records.append(dict(voice=voice,command_voice=s["voice"],command=s["carrier"],room=s["room"],
                        role=role,cue_index=index,source_pcm_sha256=source["pcm_sha256"],raw=raw,mask=masked))
        for p in sorted(work.glob("*-attempt*.gz")):
            logs.write(p,"retry/"+p.name)
    assert len(records)==256
    with ZipFile(tmp) as logs:
        assert logs.testzip() is None
        for cap in captures:
            j=cap["id"]
            data = logs.read(f"opus/{j:03}-voice.log").decode().replace("\r\n","\n")
            for lane,name in enumerate(("selected","raw")):
                packets=gap.uplink.packets(data,j,cap["emitted"]-cap["first"],lane)
                assert ch.digest(gap.uplink.decode(decoder,packets))==cap["lanes"][name]["decoded_sha256"]
    tmp.replace(pack)
    ch.save(OUT,dict(records=records,captures=captures,summary=summary,archive_sha256=ch.sha(pack),
        asr=identity,firmware=fw, sources={p.name:ch.sha(p) for p in (prior.OUT,prior.selection.FIT,fit.FIT,fit.WEIGHTS,Path(__file__))},
        policy="All frozen per-voice selections on all sixteen previously exposed room scenes, each with its same-voice Nick. "
        "Source PCM reproduces archived input byte-for-byte. Score original C161 and model-enhanced C161 at the "
        "first usable input block. Full synthetic processed signal sent to P4 Opus if a target opens."))


if __name__ == "__main__":
    main()
