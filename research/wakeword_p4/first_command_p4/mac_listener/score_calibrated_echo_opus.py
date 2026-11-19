"""Encode time-matched calibrated and original captures with the frozen panel Opus firmware."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import torch

import coded_command_uplink as uplink
import time_calibrated_echo_names as timing

ch = timing.ch
OUT = ch.RESULTS / (timing.STEM + "-opus.json")
ARCHIVE = OUT.with_suffix(".zip")
TARGETS = ("Hey Nyx", "Hey Nicks")


def capture(work, logs, decoder, reader, mic, residual, row, first, index):
    end = row["end"]
    clean, raw = residual[first:end], mic[first:end]
    count = len(clean)
    payload = struct.pack("<3I", 0x3552504f, index, count) + clean.tobytes() + raw.tobytes()
    path = work / f"capture-{index:03}.log"
    uplink.boot(uplink.CODEC, path, b"UPLINK PASS", payload, work)
    text = path.read_text()
    logs.write(path, path.name)
    result = dict(index=index, first=first, end=end, count=count, uart=path.name, uart_sha256=ch.sha(path), lanes={})
    path.unlink()
    for lane, (name, pcm) in enumerate((("residual", clean), ("raw", raw))):
        packets = uplink.packets(text, index, count, lane)
        decoded = uplink.decode(decoder, packets)
        reading = uplink.recognition(reader, decoded, row["command"])
        result["lanes"][name] = dict(input_sha256=ch.digest(pcm), decoded_sha256=ch.digest(decoded),
                                     packets=len(packets), text=reading["text"], complete=reading["complete_suffix"])
    return result


def main():
    import hashlib
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    fw = uplink.firmware()
    parent, measurement = ch.read(timing.names.OUT), ch.read(timing.OUT)
    assert measurement["input_sha256"] == ch.sha(timing.names.OUT)
    reader, identity = ch.load_reader()
    assert parent["asr"] == identity
    decoder = uplink.decoder_api()
    records, captures, summary = [], [], defaultdict(lambda: defaultdict(Counter))
    with TemporaryDirectory(dir=timing.WORK) as folder, ZipFile(ARCHIVE.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        uplink.WORK = work
        for (row, mic, residual), timed in zip(timing.signals(parent), measurement["records"], strict=True):
            assert timed["mic_sha256"] == row["mic_sha256"] and timed["residual_sha256"] == ch.digest(residual)
            if row["role"] not in TARGETS:
                continue
            old = row["stages"]["filtered"]
            routes = {"prior_p95":old["estimated"], "standard":timed["standard"]["events"], "early":timed["early"]["events"]}
            case = dict(ordinal=timed["ordinal"], role=row["role"], routes={})
            per_start = {}
            for mode, events in routes.items():
                total = summary[row["role"]][mode]
                total["trials"] += 1
                hits = [t for t in events if row["start"] <= t < row["end"]]
                if not hits:
                    case["routes"][mode] = None
                    continue
                opened = hits[0]
                # Preserve the previous host PCM capture contract to isolate the Opus comparison.
                first = max(0, ((opened+159)//160)*160 - 32000)
                if first not in per_start:
                    result = capture(work, logs, decoder, reader, mic, residual, row, first, len(captures))
                    per_start[first] = result
                    captures.append(result)
                result = per_start[first]
                clean = result["lanes"]["residual"]
                if mode == "prior_p95":
                    assert (first, clean["input_sha256"]) == (old["first"], old["capture_sha256"])
                before = opened < row["command_at"]
                total.update(dict(opened=1, before_command=before, complete=clean["complete"],
                                  timely_complete=before and clean["complete"], raw_complete=result["lanes"]["raw"]["complete"]))
                case["routes"][mode] = dict(clock=opened, capture=result["index"])
            records.append(case)
            if len(records) % 12 == 0:
                print(len(records), "target conditions", len(captures), "captures", flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for f in retries:
            logs.write(f, f.name)
    assert len(records) == 72
    with ZipFile(ARCHIVE.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for cap in captures:
            data = logs.read(cap["uart"])
            assert hashlib.sha256(data).hexdigest() == cap["uart_sha256"]
            text = data.decode().replace("\r\n", "\n")
            for lane, name in enumerate(("residual", "raw")):
                packets = uplink.packets(text, cap["index"], cap["count"], lane)
                assert ch.digest(uplink.decode(decoder, packets)) == cap["lanes"][name]["decoded_sha256"]
    ARCHIVE.with_suffix(".tmp").replace(ARCHIVE)
    ch.save(OUT, dict(records=records, captures=captures, summary=summary, archive_sha256=ch.sha(ARCHIVE),
            asr=identity, opus_sha256=fw["opus_sha256"], qemu_sha256=fw["qemu_sha256"],
            decoder=decoder.opus_get_version_string().decode(), timing_sha256=ch.sha(timing.OUT),
            script_sha256=ch.sha(Path(__file__)), retries=[f.name for f in retries],
            policy="All 72 Nyx and Nicks target conditions. Independent P4 Opus guest encodes calibrated residual "
            "and time-matched raw mixture. For each clock use the previous report's two-second history from the "
            "next 10 ms frame boundary through the same labelled end. Score unprompted small.en complete suffix "
            "on full decoded audio including codec delay. Hash and decode archived packets twice. "
            "The inferred OPEN times and fixed capture endpoint are simulated, without network transport or VAD."))


if __name__ == "__main__":
    main()
