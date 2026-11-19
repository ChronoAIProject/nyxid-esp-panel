"""Compare post-FINISH UART export on the same 24 af_bella P4 simulator inputs."""
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
import threading
import time
from zipfile import ZIP_DEFLATED, ZipFile

import replay_bella_command_prefix as bella
import replay
from transition import samples

ROOT = Path(".local/wakeword")
BUILDS = {"original": ROOT / "c161-panel-middle-confirmation/build",
          "paced": ROOT / "c161-panel-middle-uart/build"}
OUT = bella.ch.RESULTS / "4090-c161-bella-uart-watchdog.json"
RAW = ROOT / "c161-panel-middle-uart" / OUT.with_suffix(".zip").name
EVENTS = ("READY ", "HITS ", "CUE_ROUTE ", "C161_ALLOW ", "V5_ALLOW ", "OPEN ", "LISTENER PASS")


def watchdog_events(text):
    warnings = []
    for match in re.finditer(r"Task watchdog got triggered\.", text):
        pos = match.start()
        blocks = re.findall(r"^(AUDIO|PROCESSED) (\d+) ", text[:pos], re.M)
        warnings.append(dict(after_finish=text.index("FINISH ") < pos < text.index("LISTENER PASS"),
                             last_pcm_record=blocks[-1] if blocks else None))
    return warnings


def observe(log, old, events):
    text = log.read_text()
    assert [s for s in text.splitlines() if s.startswith(EVENTS)] == events
    finish = replay.values(text, "FINISH")
    assert finish[:4] == old["finish"][:4]
    digests = {tag: replay.digest(samples(text, tag, finish[2])) for tag in ("AUDIO", "PROCESSED")}
    assert digests["AUDIO"] == old["uplink"]["fifo_sha256"]
    return dict(finish=finish, watchdog=watchdog_events(text), pcm_sha256=digests, uart_sha256=replay.sha(log))


def long_export(work, archive, old, mic):
    # Extend the first historical warning case to 20 s with digital silence.
    # Its entire original command and independently dated panel event stay in place.
    length = 320000
    idx = old["index"]
    assert len(mic) < length and replay.digest(mic) == old["mic_sha256"]
    signal = mic.tobytes() + bytes((length-len(mic))*2)
    payload = (struct.pack("<3I", 0x5854594e, idx, length) + signal + bytes(length*2)
               + struct.pack("<I", old["source"]["transmit_event"]))
    row = dict(parent_index=idx, input_samples=length, mic_sha256=sha256(signal).hexdigest())
    for variant, build in BUILDS.items():
        log = work / "serial.log"
        replay.boot(build, log, work / "flash.bin", b"LISTENER PASS", (2 << 20, payload))
        archive.write(log, f"{variant}/long-{idx:03}.log")
        text = log.read_text()
        finish = replay.values(text, "FINISH")
        assert finish[:2] == [length, old["finish"][1]]
        assert replay.values(text, "OPEN")[0] == old["opened"][0]
        selected = samples(text, "AUDIO", finish[2])
        assert replay.digest(selected[:old["finish"][2]]) == old["uplink"]["fifo_sha256"]
        row[variant] = dict(finish=finish, watchdog=watchdog_events(text), uart_sha256=replay.sha(log),
                            events=[s for s in text.splitlines() if s.startswith(EVENTS)],
                            pcm_sha256=dict(AUDIO=replay.digest(selected),
                                            PROCESSED=replay.digest(samples(text, "PROCESSED", finish[2]))))
    for key in ("events", "pcm_sha256"):
        assert row["original"][key] == row["paced"][key]
    return row


def slow_export(work, archive, old, mic, events):
    # Drain the emulator UART through a bounded pipe. The host rate is a test
    # condition, independent of the board's configured baud rate.
    rate = 96000
    idx = old["index"]
    assert replay.digest(mic) == old["mic_sha256"]
    payload = (struct.pack("<3I", 0x5854594e, idx, len(mic)) + mic.tobytes() + bytes(mic.nbytes)
               + struct.pack("<I", old["source"]["transmit_event"]))
    row = dict(parent_index=idx, max_host_bytes_per_second=rate, mic_sha256=old["mic_sha256"])
    for variant, build in BUILDS.items():
        log = work / "serial.log"
        fifo = work / f"{variant}.fifo"
        os.mkfifo(fifo)
        stop = threading.Event()

        def drain():
            with fifo.open("rb", buffering=0) as source, log.open("wb", buffering=0) as out:
                while not stop.is_set():
                    chunk = source.read(2048)
                    if not chunk:
                        break
                    out.write(chunk)
                    time.sleep(len(chunk) / rate)

        pump = threading.Thread(target=drain, daemon=True)
        pump.start()
        try:
            replay.boot(build, log, work / "flash.bin", b"LISTENER PASS", (2 << 20, payload), serial_target=fifo)
        finally:
            stop.set()
            pump.join(timeout=2)
        archive.write(log, f"{variant}/slow-{idx:03}.log")
        row[variant] = observe(log, old, events)
    assert row["original"]["pcm_sha256"] == row["paced"]["pcm_sha256"]
    return row


def main():
    baseline = replay.read(bella.OUT)
    firmware = {name: replay.sha(b / "nyx_mac_listener.bin") for name, b in BUILDS.items()}
    assert firmware["original"] == baseline["firmware"] and firmware["paced"] != firmware["original"]
    model = (replay.WORK / "model/hey_nyx_model.bin").read_bytes()
    assert replay.sha(replay.WORK / "model/hey_nyx_model.bin") == replay.C161
    configs = [replay.read(b / "config/sdkconfig.json") for b in BUILDS.values()]
    for b in BUILDS.values():
        assert (b / "nyx_mac_listener.bin").read_bytes().count(model) == 1
    assert configs[0] == configs[1]
    assert configs[0]["ESP_TASK_WDT_CHECK_IDLE_TASK_CPU0"] and configs[0]["ESP_TASK_WDT_TIMEOUT_S"] == 5
    assert replay.sha(replay.QEMU) == baseline["codec"]["qemu_sha256"]
    previous_zip = bella.OUT.with_suffix(".zip")
    assert replay.sha(previous_zip) == baseline["archive_sha256"]
    rows = []
    with TemporaryDirectory(dir=RAW.parent) as tmp, ZipFile(previous_zip) as previous:
        work = Path(tmp)
        with ZipFile(work / RAW.name, "w", ZIP_DEFLATED) as raw:
            extended = None
            for parent, source, mic, _, _ in bella.recordings():
                if not parent["paired_back_grid"]:
                    continue
                idx = len(rows)
                old = baseline["captures"][idx]
                assert old["source"] == source and old["batch"] == parent["batch"]
                assert replay.digest(mic) == old["mic_sha256"]
                reference = bytes(mic.nbytes)
                assert sha256(reference).hexdigest() == old["reference_sha256"]
                payload = (struct.pack("<3I", 0x5854594e, idx, len(mic)) + mic.tobytes()
                           + reference + struct.pack("<I", source["transmit_event"]))
                events = [line for line in previous.read(f"events/{idx:03}.txt").decode().splitlines()
                          if line.startswith(EVENTS)]
                row = dict(index=idx, file=f'{old["batch"]}/{source["index"]:03}.wav',
                           opened=old["opened"], mic_sha256=old["mic_sha256"], events=events,
                           historical_watchdog=f"watchdog/{idx:03}.txt" in previous.namelist())
                order = list(BUILDS) if idx % 2 == 0 else list(reversed(BUILDS))
                for variant in order:
                    log = work / "serial.log"
                    replay.boot(BUILDS[variant], log, work / "flash.bin", b"LISTENER PASS", (2 << 20, payload))
                    raw.write(log, f"{variant}/{idx:03}.log")
                    row[variant] = observe(log, old, events)
                assert row["original"]["pcm_sha256"] == row["paced"]["pcm_sha256"]
                rows.append(row)
                if extended is None and row["historical_watchdog"]:
                    extended = old, mic.copy()
                print(idx+1, "of 24", *(len(row[v]["watchdog"]) for v in BUILDS), flush=True)
            stress = long_export(work, raw, *extended)
            slow = slow_export(work, raw, *extended, rows[extended[0]["index"]]["events"])
        assert len(rows) == 24
        with ZipFile(work / RAW.name) as raw:
            assert len(raw.namelist()) == 52 and raw.testzip() is None
        report = dict(firmware=firmware, model=replay.C161, qemu=baseline["codec"]["qemu_sha256"],
                      wdt_timeout_seconds=configs[0]["ESP_TASK_WDT_TIMEOUT_S"], idle_check_cpu0=True,
                      baseline_sha256=replay.sha(bella.OUT), raw=RAW.name, raw_sha256=replay.sha(work / RAW.name),
                      raw_bytes=(work / RAW.name).stat().st_size, records=rows, long_export=stress,
                      slow_export=slow)
        (work / OUT.name).write_text(json.dumps(report, indent=2)+"\n")
        (work / RAW.name).replace(RAW)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
