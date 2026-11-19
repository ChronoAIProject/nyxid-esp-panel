"""Evaluate a bounded C161 wait when the first reference marker predicts an imminent pair."""
from bisect import bisect_left, bisect_right
import argparse
import ctypes
import re
import struct
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import handoff_clock_rates as rates
from code_gain_guard import activity_pulses
from prefix_guard import collisions


ch, handoff = rates.ch, rates.handoff
PORTABLE = Path(__file__).resolve().parents[2]/"portable"
PAIR = PORTABLE/"c161_coded_cue_aligned_v4.h"
OUT = ch.RESULTS/"4090-c161-first-reference-wait.json"
PAIR_GAP = 2560
TOLERANCE = 24
MAX_EXTENSION = 768


def inspect_first(work):
    # Instrument the frozen matcher's existing dot-product assignment in a temporary copy.
    header = PAIR.read_text()
    assignment = "first_[slot_] = sign(0);"
    assert header.count(assignment) == 1
    instrument = "thread_local bool observed_first;\n" + header.replace(
        assignment, assignment + " observed_first = first_[slot_] != 0;")
    wrapper = """
extern "C" void inspect(const int16_t* ref, int samples, uint8_t* events) {
    nyx::AlignedCodedCueV4 pair;
    for (int i = 0; i < samples; ++i) {
        observed_first = false;
        const bool second = pair.update(ref[i]);
        events[i] = (second ? 2 : 0) + (observed_first ? 1 : 0);
    }
}
"""
    (work/"observed.h").write_text(instrument)
    (work/"probe.cpp").write_text('#include "observed.h"\n' + wrapper)
    lib = work/"probe.so"
    subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(PORTABLE),
                    str(work/"probe.cpp"), "-o", str(lib)], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.inspect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    dll.inspect.restype = None
    return dll


def first_events(dll, reference):
    ref = np.ascontiguousarray(reference, dtype="<i2")
    bits = np.zeros(len(ref), np.uint8)
    dll.inspect(ref.ctypes.data, len(ref), bits.ctypes.data)
    return [(np.flatnonzero(bits & flag)+1).tolist() for flag in (1, 2)]


def allowance(at, active, first, events, adaptive):
    i = bisect_right(active, at)
    playing = i > 0 and active[i-1] >= at-8000
    due = at + (480 if playing else 0)
    nominal = due
    last = None
    if playing and adaptive:
        # At the original deadline inspect only previous first markers. Expire once even
        # if unrelated speech produces new markers while waiting. A lone marker cannot veto.
        lo = bisect_left(first, due - PAIR_GAP - TOLERANCE)
        hi = bisect_right(first, due - PAIR_GAP - TOLERANCE - 1 + MAX_EXTENSION)
        if hi > lo:
            last = first[hi-1]
            due = last + PAIR_GAP + TOLERANCE + 1
    prefix = set(events["qualified_prefix"])
    for t in events["prefix_reference"]:
        if t not in prefix and t <= due <= t+1944:
            due = t+1945
    blocked = any(t <= due and t+3488 >= at for t in events["reference_events"])
    blocked |= any(t <= due and t+1944 >= at for t in prefix)
    return dict(hit=at, nominal_due=nominal, first_marker=last, due=due, allowed=None if blocked else due)


def compare(dll, mic, ref, old, meta):
    assert rates.listener.digest(mic) == old["mic_sha256"] and rates.listener.digest(ref) == old["reference_sha256"]
    first, pairs = first_events(dll, ref)
    assert pairs == old["prefix_reference"]
    active = (np.flatnonzero(activity_pulses(ref))+1).tolist()
    fixed = [allowance(t, active, first, old, False) for t in old["fixed"]]
    model = [allowance(t, active, first, old, True) for t in old["c161"]]
    base = [allowance(t, active, first, old, False) for t in old["c161"]]
    def allowed(rows):
        return sorted(x["allowed"] for x in rows if x["allowed"] is not None)
    assert allowed(base+fixed) == old["allowed"]
    result = allowed(model+fixed)
    opened = [(result[0]+255)//256*256] if result else []
    return dict(source=meta, mic_sha256=old["mic_sha256"], reference_sha256=old["reference_sha256"],
                first_count=len(first), reference_pairs=pairs, reference_full=old["reference_events"],
                qualified_prefix=old["qualified_prefix"], fixed=old["fixed"], c161=model,
                allowed_fixed=[x["allowed"] for x in fixed if x["allowed"] is not None],
                old_opened=old["opened"], opened=opened)


def cases(work):
    drift = [r for r in ch.read(rates.STEM.with_suffix(".json"))["records"] if r["player"] == "independent_reference"]
    for (meta, mic, ref, _), old in zip(rates.independent_reference(), drift, strict=True):
        assert old["source"] == meta
        yield "drift", meta, mic, ref, old
    for group, stream in (("transitions", handoff.test.transitions), ("rooms", handoff.test.rooms)):
        path = handoff.test.BASE.with_name(handoff.test.BASE.name+"-"+group+".json")
        for (meta, mic, ref, _), old in zip(stream(), ch.read(path)["records"], strict=True):
            assert meta == old["source"]
            yield group, meta, mic, ref, old
    full = ch.binding(work/"full.so", coded="v5")
    pair = ch.binding(work/"pair.so", coded="aligned")
    native = ch.c161_binding()
    for meta, mic, ref, _ in collisions():
        routes = ch.routes(full, native, mic)
        old = rates.guarded(routes, full, pair, ref)
        yield "speech_collision", meta, mic, ref, old
    # One first code on the reference, with the same unrelated external positive on the mic.
    at, = routes["c161"]
    carrier = ch.read(rates.listener.rooms.PLAN)["records"][meta["mic_index"]]["carrier"]
    info = ch.read(rates.fixed.SOURCE_INFO)
    cue, rate = sf.read(rates.fixed.CUE, dtype="int16")
    assert rate == 16000 and ch.sha(rates.fixed.CUE) == info["wav_sha256"]
    start, length = info["code_positions"][0], info["code_length"]
    code = cue[start:start+length]
    for ahead in (0, 256, 512, 768, 1024):
        ref = np.zeros_like(mic)
        pos = at + 480 + ahead - PAIR_GAP - length
        ref[pos:pos+length] = code
        old = rates.guarded(routes, full, pair, ref)
        assert not old["reference_events"] and not old["prefix_reference"] and old["opened"]
        meta = dict(ref_code_start=pos, model_at=at, nominal_second_minus_due=ahead, carrier=carrier)
        yield "isolated_first", meta, mic, ref, old


def host():
    sources = [PAIR, Path(__file__), rates.STEM.with_suffix(".json")]
    sources.extend(handoff.test.BASE.with_name(handoff.test.BASE.name+"-"+g+".json") for g in ("rooms", "transitions"))
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        dll = inspect_first(work)
        rows = [dict(group=group, **compare(dll, mic, ref, old, meta)) for group,meta,mic,ref,old in cases(work)]
    assert len(rows) == 88
    ch.save(OUT, dict(records=rows, source_sha256={p.name:ch.sha(p) for p in sources},
        max_extra_samples=MAX_EXTENSION, second_after_first_samples=PAIR_GAP, tolerance_samples=TOLERANCE,
        policy="Host counterfactual using the frozen pair matcher's first correlations, playback activity and detector "
        "clocks. Only a pending C161 decision may wait once for a predicted second marker within 48 ms. Once the "
        "second marker arrives the existing signature/third-marker rule applies. Isolated-first controls use a "
        "genuine external C161 hit but an artificial electrical reference."))


def p4():
    listener = rates.listener
    build = listener.WORK/"first-reference-wait/build"
    binary = build/"nyx_mac_listener.bin"
    model = listener.WORK/"model/hey_nyx_model.bin"
    assert ch.sha(model) == listener.C161 and binary.read_bytes().count(model.read_bytes()) == 1
    grid = ch.read(OUT)
    assert grid["source_sha256"][Path(__file__).name] == ch.sha(Path(__file__))
    reader, identity = ch.load_reader()
    decoder = rates.fixed.uplink.decoder_api()
    base = OUT.with_name(OUT.stem+"-p4")
    archive = base.with_suffix(".zip")
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        for i, ((group, meta, mic, ref, old), host_row) in enumerate(zip(cases(work), grid["records"], strict=True)):
            assert (group, meta, ch.digest(mic), ch.digest(ref)) == (
                host_row["group"], host_row["source"], host_row["mic_sha256"], host_row["reference_sha256"])
            row = dict(group=group, source=meta)
            for mode in (("original", "candidate") if group == "isolated_first" else ("candidate",)):
                selected = mode == "candidate"
                idx = i if selected else 100+i
                protocol = 0x5752594e if selected else 0x4652594e
                payload = struct.pack("<3I", protocol, idx, len(mic)) + mic.tobytes() + ref.tobytes()
                log = work/"serial.log"
                listener.boot(build, log, work/"image.bin", b"LISTENER PASS", (2<<20, payload))
                uart = log.read_text()
                assert listener.values(uart, "READY") == [idx, len(mic), 256, 81600]
                hits = [tuple(map(int,s.split())) for s in re.findall(r"^HITS (.+)$", uart, re.M)]
                assert [c for _,c,_ in hits if c >= 0] == old["c161"]
                assert [c for _,_,c in hits if c >= 0] == old["fixed"]
                opens = [int(t) for t in re.findall(r"^OPEN (\d+) ", uart, re.M)]
                assert opens == (host_row["opened"] if selected else old["opened"])
                if selected:
                    assert [int(t) for t in re.findall(r"^C161_ALLOW (\d+)$", uart, re.M)] == [
                        e["allowed"] for e in host_row["c161"] if e["allowed"] is not None]
                    assert [int(t) for t in re.findall(r"^V5_ALLOW (\d+)$", uart, re.M)] == host_row["allowed_fixed"]
                result = dict(opened=opens, finish=listener.values(uart, "FINISH"), uart_sha256=ch.sha(log))
                tags = ("READY ", "HITS ", "C161_ALLOW ", "V5_ALLOW ", "OPEN ", "FINISH ", "LISTENER PASS")
                logs.writestr(f"{idx:02}-listener.log", "\n".join(s for s in uart.splitlines() if s.startswith(tags))+"\n")
                if group == "isolated_first":
                    result["uplink"] = handoff.bench.fifo(work, logs, idx, mic, ref, dict(source=meta, opened=opens),
                                                         reader, decoder, True, True)
                elif "uplink" in old and opens == old["opened"]:
                    from transition import samples
                    fifo = samples(uart, "AUDIO", result["finish"][2])
                    result["fifo_sha256"] = ch.digest(fifo)
                    assert result["fifo_sha256"] == old["uplink"]["fifo_sha256"]
                row[mode] = result
            rows.append(row)
            if (i+1)%10 == 0:
                print(i+1, group, flush=True)
    assert len(rows) == len(grid["records"]) == 88
    archive.with_suffix(".tmp").replace(archive)
    script = Path(__file__)
    ch.save(base.with_suffix(".json"), dict(records=rows, asr=identity, grid_sha256=ch.sha(OUT),
        archive_sha256=ch.sha(archive), listener_sha256=ch.sha(binary), model_sha256=ch.sha(model),
        codec_sha256=ch.sha(rates.fixed.uplink.CODEC/"nyx_first_command.bin"), qemu_sha256=ch.sha(listener.QEMU),
        sources={p.name: ch.sha(p) for p in (script, script.with_name("first_reference_wait.patch"),
                 script.with_name("build_first_reference_wait.sh"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("host", "p4"))
    {"host":host, "p4":p4}[parser.parse_args().action]()
