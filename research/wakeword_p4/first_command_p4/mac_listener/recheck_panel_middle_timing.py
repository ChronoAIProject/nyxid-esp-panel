"""Replay every over-budget input in the single-C firmware matrix on identical bytes."""
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import check_panel_middle_p4 as middle

ch, tx = middle.ch, middle.tx
OUT = middle.OUT.with_name("4090-c161-panel-middle-timing-recheck.json")


def main():
    report = ch.read(middle.OUT)
    build = middle.WORK / "build"
    assert ch.sha(build / "nyx_mac_listener.bin") == report["firmware"]
    archive = middle.OUT.with_suffix(".zip")
    assert ch.sha(archive) == report["archive_sha256"]
    originals = {r["host_index"]: r for r in ch.read(tx.diag.BOARD)["records"]
                 if r["name"] == "bca" and r["role"] == "both"}
    inputs = []
    with ZipFile(archive) as logs:
        for row in report["records"]:
            if not row["finish"][6]:
                continue
            assert row["group"].startswith("balanced.")
            _, mic, _, _, _ = next(tx.balanced.inputs(originals[row["host_index"]]))
            ref = np.zeros_like(mic)
            assert (ch.digest(mic), ch.digest(ref)) == (row["mic_sha256"], row["reference_sha256"])
            index = row["index"]
            payload = (struct.pack("<3I", 0x5854594e, index, len(mic)) + mic.tobytes()
                       + ref.tobytes() + struct.pack("<I", row["transmit_event"]))
            events = logs.read(f"events/{index:03}.txt").decode().splitlines()
            inputs.append((row, payload, events))

    results = []
    listener = tx.panel.phase.listener
    with TemporaryDirectory() as folder:
        work = Path(folder)
        serial = work / "serial.log"
        for repeat in range(3):
            for row, payload, events in inputs:
                listener.boot(build, serial, work / "image.bin", b"LISTENER PASS", (2 << 20, payload))
                uart = serial.read_text()
                measured = listener.values(uart, "FINISH")
                prefixes = ("READY ", "HITS ", "V5_ALLOW ", "C161_ALLOW ", "OPEN ", "CUE_ROUTE ", "LISTENER PASS")
                assert [s for s in uart.splitlines() if s.startswith(prefixes)] == [
                    s for s in events if s.startswith(prefixes)]
                assert measured[:4] == row["finish"][:4]
                results.append(dict(index=row["index"], group=row["group"], repeat=repeat,
                                    original_finish=row["finish"], finish=measured, uart_sha256=ch.sha(serial)))
    ch.save(OUT, dict(records=results, firmware=report["firmware"], sources={
        p.name: ch.sha(p) for p in (middle.OUT, tx.diag.BOARD, Path(__file__))},
        policy="Replay each over-budget source of the single-C matrix three times on identical M, R and TX. "
               "Check event clocks, routing and emitted FIFO boundaries. Timing is ESP32-P4 QEMU wall time."))


if __name__ == "__main__":
    main()
