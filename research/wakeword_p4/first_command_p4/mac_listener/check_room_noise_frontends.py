"""Measure wake and complete-command behavior after single-microphone noise suppression."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_c161_v5_command_deepfilter as deep
import check_c161_v5_command_ns as ns
import diagnose_lewis_room_misses as room

ch, gap = room.ch, room.gap
OUT = ch.RESULTS / "4090-c161-lewis-room-noise-frontends.json"


def inputs():
    nyx, rate = sf.read(gap.CUE, dtype="int16")
    assert rate == 16000
    nick, = [p for r,p in gap.study.prefixes() if (
        r["voice"], r["style"], r["role"]) == ("bm_lewis", "connected", "Hey Nick")]
    for s, old, kind, h, bg, cmd in room.room_inputs():
        for role, cue in (("lewis_nyx",nyx), ("lewis_nick",np.pad(nick,(0,gap.GAP))),
                          ("silence",np.zeros_like(nyx))):
            start = s["end"]-len(cue)
            signal = bg+cmd
            wet = fftconvolve(cue.astype(float), h)
            signal[start:start+len(wet)] += wet
            pcm = room.quantize(signal*s["gain"])
            expected = s["pcm_sha256"] if role == "silence" else old["routes"][role]["pcm_sha256"]
            assert ch.digest(pcm) == expected
            yield dict(role=role, start=start, end=s["end"], command=s["carrier"], command_voice=s["voice"],
                       ns_output_start=0, kind=kind, pcm_sha256=expected), pcm


def main():
    parent = ch.read(ch.RESULTS/(ns.STEM+".json"))
    assert ch.sha(ns.BUILD/"nyx_first_command.elf") == parent["elf_sha256"]
    assert ch.sha(ns.QEMU) == parent["qemu_sha256"] and ch.sha(deep.BINARY) == deep.BINARY_SHA
    fw = gap.uplink.firmware()
    decoder = gap.uplink.decoder_api()
    reader, asr = ch.load_reader()
    native = room.rooms.c161_binding()
    rows = list(inputs())
    records, captures = [], []
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(Counter)))
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        ns.replay(rows, work)
        for group in range(0,len(rows),3):
            log = work/f"{group}.log"
            logs.write(log, "ns/"+log.name)
            for offset, (origin, filtered, allocated) in enumerate(ns.frames(log, rows, group)):
                s, pcm = rows[group+offset]
                assert origin == 0
                restored, clips = deep.enhance(pcm,work)
                entry = dict(source=s, allocated=allocated, deep_clips=clips, ns_uart_sha256=ch.sha(log), routes={})
                for name, y in (("raw",pcm), ("p4_ns",filtered[:len(pcm)]), ("deep",restored)):
                    result = room.measure(native, y, s["start"], s["end"])
                    events = [gap.study.voices.timing.block(t) for t in result["events"]]
                    hits = [t for t in events if s["start"] <= t < len(pcm)]
                    result.update(pcm_sha256=ch.digest(y), delivered=events)
                    stats = summary[s["kind"]][s["role"]][name]
                    stats.update(trials=1, hits=bool(hits), timely=bool(hits and hits[0] < s["end"]),
                                 pre_cue=sum(t < s["start"] for t in events))
                    if hits and s["role"] == "lewis_nyx":
                        i = len(captures)
                        cap = gap.replay.capture(reader, decoder, work, logs, i, "denoised", y, hits[0], s["command"])
                        captures.append(dict(id=i, input=group+offset, mode=name, **cap))
                        complete = cap["lanes"]["selected"]["asr"]["complete_suffix"]
                        stats.update(complete=complete, timely_complete=complete and hits[0] < s["end"])
                        result.update(capture=i, complete=complete)
                    entry["routes"][name] = result
                records.append(entry)
            log.unlink()
            print(len(records), len(captures), flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for p in retries:
            logs.write(p,"retry/"+p.name)
    assert len(records) == 48
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for c in captures:
            i = c["id"]
            text = logs.read(f"opus/{i:03}-denoised.log").decode().replace("\r\n", "\n")
            for lane, name in enumerate(("selected","raw")):
                packets = gap.uplink.packets(text, i, c["emitted"]-c["first"], lane)
                assert ch.digest(gap.uplink.decode(decoder, packets)) == c["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=records, captures=captures, summary=summary, asr=asr, firmware=fw,
        ns_elf_sha256=parent["elf_sha256"], deep_binary_sha256=ch.sha(deep.BINARY), archive_sha256=ch.sha(archive),
        sources={p.name:ch.sha(p) for p in (room.parent.OUT, gap.CUE, Path(ns.__file__), Path(deep.__file__),
                Path(room.__file__), Path(__file__))}, retries=[p.name for p in retries],
        policy="Run full raw mix, actual causal P4 NS output (10 ms frames, medium mode, no delay compensation), "
        "or released host DeepFilterNet with latency compensation from the start of each input. Frozen C161 "
        "alone; record delivered block clock and P4 Opus processed waveform for every detected target. The host "
        "DeepFilterNet alignment is an offline upper-bound timing assumption. No joint firmware scheduling, "
        "physical acoustic measurements or independent source voices."))


if __name__ == "__main__":
    main()
