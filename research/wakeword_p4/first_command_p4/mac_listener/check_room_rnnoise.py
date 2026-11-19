"""Replay pinned RNNoise on a simulated P4 and score the complete causal output."""
import argparse
from collections import Counter, defaultdict
import gzip
from pathlib import Path
import struct
import time
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import check_room_noise_frontends as parent
import prepare_reusable_nyx as quiet
from run_c161_first_command_v5 import QEMU, boot

ch, room, gap = parent.ch, parent.room, parent.gap
WORK = Path(".local/wakeword/c161-rnnoise")
BUILD = WORK / "build"
OUT = ch.RESULTS / "4090-c161-room-rnnoise-p4.json"
ELF = BUILD / "nyx_nsnet_probe.elf"
HOP = 160


def inputs():
    yield from parent.inputs()
    for s, clip in quiet.prefixes():
        pcm = np.zeros(80000, dtype="<i2")
        start = 48000 - len(clip)
        pcm[start:48000] = clip
        yield dict(kind="quiet", role=s["role"], start=start, end=48000, voice=s["voice"],
                   style=s["style"], command=None, pcm_sha256=ch.digest(pcm)), pcm


def parse(data, i, n):
    timings, raw = [], bytearray()
    ready, active, finish = [], [], []
    for line in data.decode("ascii", errors="replace").splitlines():
        if line.startswith("READY "):
            ready.append(list(map(int, line.split()[1:])))
        elif line.startswith("FRAME "):
            at, status, usec = map(int, line.split()[1:])
            assert at == len(timings) * HOP == len(raw) // 2 and status == 0 and usec > 0
            timings.append(usec)
        elif line.startswith("PCM "):
            _, at, hexdump = line.split()
            chunk = bytes.fromhex(hexdump)
            assert int(at) == len(raw) // 2 == (len(timings) - 1) * HOP and len(chunk) == 2 * HOP
            raw.extend(chunk)
        elif line.startswith("ACTIVE "):
            active.append(list(map(int, line.split()[1:])))
        elif line.startswith("FINISH "):
            finish.append(list(map(int, line.split()[1:])))
    assert len(ready) == len(active) == len(finish) == 1 and data.count(b"NSNET PASS") == 1
    assert ready[0][:4] == [i, n, HOP, 16000] and finish[0][0] == i
    assert len(raw) == 2 * HOP * ((n + HOP - 1) // HOP)
    return np.frombuffer(raw, "<i2")[:n], timings, dict(
        create_internal=ready[0][4], create_external=ready[0][5], active_internal_free=active[0][0],
        boot_min_internal_free=active[0][1], stack_min_free=active[0][2],
        finish_internal_free=finish[0][1], finish_external_free=finish[0][2])


def cache_file(i, src, firmware):
    return WORK / "uart" / firmware / f'{i:02}-{src["pcm_sha256"]}.gz'


def collect(rows=None, first=0):
    firmware = ch.sha(ELF)
    for i, (s, pcm) in enumerate(inputs() if rows is None else rows, first):
        dest = cache_file(i, s, firmware)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            parse(gzip.decompress(dest.read_bytes()), i, len(pcm))
            continue
        payload = struct.pack("<3I", 0x324e534e, i, len(pcm)) + pcm.astype("<i2", copy=False).tobytes()
        log = WORK / f"current-{first}.log"
        for attempt in range(3):
            try:
                boot(BUILD, log, WORK / f"scene-flash-{first}.bin", b"NSNET PASS", (2 << 20, payload))
                data = log.read_bytes()
                parse(data, i, len(pcm))
                break
            except (AssertionError, TimeoutError):
                dest.with_suffix(f".attempt-{attempt+1}.gz").write_bytes(gzip.compress(log.read_bytes()))
                if attempt == 2:
                    raise
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(data, compresslevel=3))
        tmp.replace(dest)
        print(i, s["kind"], s["role"], flush=True)


def service_stats(us):
    return dict(frames=len(us), median_us=float(np.median(us)), p95_us=float(np.percentile(us, 95)),
                max_us=max(us), above_10ms=sum(t > 10000 for t in us))


def read_block(at):
    return (at + 255) // 256 * 256


def wall_clock(events, costs):
    # Every input block has 256 samples. RNNoise releases 160 outputs at once.
    # Use microseconds times two to represent the 62.5 us sample duration exactly.
    ready, done = [], 0
    for index, cost in enumerate(costs, 1):
        done = max(done, read_block(index * HOP) * 125) + 2 * cost
        ready.append((done + 124) // 125)
    return [ready[(event - 1) // HOP] for event in events if event > 0]


def score():
    old = ch.read(parent.OUT)
    firmware = ch.sha(ELF)
    fw = gap.uplink.firmware()
    assert fw["qemu_sha256"] == ch.sha(QEMU)
    model = room.rooms.MODEL
    model_sha = ch.sha(model)
    assert model.stat().st_size == 60952 and model_sha == "c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81"
    native = room.rooms.c161_binding()
    reader, asr = ch.load_reader()
    assert asr == old["asr"]
    decoder = gap.uplink.decoder_api()
    records, costs, cold, captures = [], [], [], []
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    archive = OUT.with_suffix(".zip")
    tmp_archive = archive.with_suffix(".tmp")
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(tmp_archive, "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for i, (s, pcm) in enumerate(inputs()):
            cached = cache_file(i, s, firmware)
            deadline = time.monotonic() + 150
            while not cached.exists() and time.monotonic() < deadline:
                time.sleep(1)
            data = gzip.decompress(cached.read_bytes())
            filtered, service, heap = parse(data, i, len(pcm))
            costs.extend(service)
            cold.append(service[0])
            logs.writestr(f"uart/{i:02}.log", data)
            row = dict(source=s, uart_sha256=ch.digest(np.frombuffer(data, dtype="u1")), samples=len(pcm),
                       heap=heap, frame_service=service_stats(service), routes={})
            for name, signal in (("raw", pcm), ("rnnoise", filtered)):
                result = room.measure(native, signal, s["start"], s["end"])
                events = result["events"]
                delivered = [read_block(t if name == "raw" else (t + HOP - 1) // HOP * HOP) for t in events]
                hits = [t for t in delivered if s["start"] <= t < len(pcm)]
                result.update(delivered=delivered, pcm_sha256=ch.digest(signal), hits=hits)
                result["timely"] = bool(hits and hits[0] < s["end"])
                stat = summary[s["kind"]][s["role"]][name]
                stat.update(trials=1, hits=bool(hits), timely=result["timely"],
                            pre_cue=sum(t < s["start"] for t in delivered))
                if name == "rnnoise":
                    with_service = wall_clock(events, service)
                    result["with_guest_service"] = with_service
                    stat.update(timely_with_guest_service=any(s["start"] <= t < s["end"] for t in with_service))
                if i < 48 and name == "raw":
                    prior = old["records"][i]
                    assert prior["source"] == s and prior["routes"]["raw"]["events"] == events
                    assert prior["routes"]["raw"]["codes_sha256"] == result["codes_sha256"]
                if hits and s["role"] == "lewis_nyx":
                    if name == "raw":
                        complete = prior["routes"]["raw"]["complete"]
                        result["prior_capture"] = prior["routes"]["raw"]["capture"]
                    else:
                        cap_id = len(captures)
                        captured = gap.replay.capture(reader, decoder, work, logs, cap_id, name,
                                                      signal, hits[0], s["command"])
                        captures.append(dict(id=cap_id, input=i, **captured))
                        result["capture"] = cap_id
                        complete = captured["lanes"]["selected"]["asr"]["complete_suffix"]
                    result["complete"] = complete
                    stat.update(complete=complete, timely_complete=complete and result["timely"])
                row["routes"][name] = result
            records.append(row)
            print(i, s["kind"], s["role"], [r["hits"] for r in row["routes"].values()], flush=True)
        for p in sorted(work.glob("*-attempt*.gz")):
            logs.write(p, "opus-retry/" + p.name)
        for p in sorted(cached.parent.glob("*.attempt-*.gz")):
            logs.write(p, "frontend-retry/" + p.name)
    assert len(records) == 64 and len(old["records"]) == 48
    with ZipFile(tmp_archive) as logs:
        assert logs.testzip() is None
        for i, row in enumerate(records):
            y, _, _ = parse(logs.read(f"uart/{i:02}.log"), i, row["samples"])
            assert ch.digest(y) == row["routes"]["rnnoise"]["pcm_sha256"]
        for cap in captures:
            j = cap["id"]
            text = logs.read(f"opus/{j:03}-rnnoise.log").decode().replace("\r\n", "\n")
            for lane, name in enumerate(("selected", "raw")):
                packets = gap.uplink.packets(text, j, cap["emitted"] - cap["first"], lane)
                assert ch.digest(gap.uplink.decode(decoder, packets)) == cap["lanes"][name]["decoded_sha256"]
    tmp_archive.replace(archive)
    app = Path("research/wakeword_p4/nsnet_p4")
    paths = [Path(__file__), Path(parent.__file__), parent.OUT, quiet.PACK, BUILD/"nyx_nsnet_probe.bin",
             WORK/"sdkconfig", *sorted(app.rglob("*.c")), *sorted(app.rglob("*.h")), *sorted(app.rglob("CMakeLists.txt"))]
    ch.save(OUT, dict(records=records, captures=captures, summary=summary, service=service_stats(costs),
        cold_frames=service_stats(cold), archive_sha256=ch.sha(archive), asr=asr, firmware=fw,
        model_sha256=model_sha, threshold=235/256, smoothing_steps=5, elf_sha256=firmware,
        rnnoise_commit="6cbfd53eb348a8d394e0757b4025c6ded34eb2b6", sources={str(p):ch.sha(p) for p in paths},
        policy="One boot per input, original C161 and exposed audio. Causal 16/48/16 kHz FIR and 10 ms RNNoise "
        "frames from sample zero. Crop only final zero padding. Release to the next 256-sample acquisition "
        "block. Opus uses that availability clock and filtered audio throughout. Separately report serialized "
        "guest service on the 256-sample capture schedule. No concurrent full-device timing. Min heap is "
        "boot-wide. Quiet uses the cue end rather than a command boundary. Raw Opus ASR is reused by hash."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("collect", "score"))
    args = parser.parse_args()
    (collect if args.step == "collect" else score)()
