"""Test the fixed first-marker wait on all twelve archived bathroom and shower self-play inputs."""
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory

import numpy as np

import probe_first_reference_wait as trial

ch, rates = trial.ch, trial.rates
OFFSETS = (-3584, -2400, 0, 1792, 2048, 2400, 2563, 3072, 3584)
BOARD_OFFSETS = (2400, 3072)
REPORT = ch.RESULTS/"4090-c161-first-reference-lag.json"


def board(work, build, protocol, index, mic, ref, expected):
    payload = struct.pack("<3I", protocol, index, len(mic)) + mic.tobytes() + ref.tobytes()
    log = work/"serial.log"
    rates.listener.boot(build, log, work/"flash.bin", b"LISTENER PASS", (2<<20, payload))
    uart = log.read_text()
    assert rates.listener.values(uart, "READY") == [index, len(mic), 256, 81600]
    opened = [int(t) for t in re.findall(r"^OPEN (\d+) ", uart, re.M)]
    assert opened == expected
    return dict(opened=opened, finish=rates.listener.values(uart, "FINISH"), uart_sha256=ch.sha(log),
                events=[s for s in uart.splitlines() if s.startswith(("HITS ", "C161_ALLOW ", "V5_ALLOW ", "OPEN "))])


def main():
    build = rates.listener.WORK/"first-reference-wait/build"
    fw = build/"nyx_mac_listener.bin"
    history = trial.OUT.with_name(trial.OUT.stem+"-p4.json")
    prior = ch.read(history)
    assert ch.sha(fw) == prior["listener_sha256"]
    witnesses = [r for r in ch.read(rates.DRIFT_BASELINE)["records"]
                 if r["group"] == "rooms" and r["source"]["player"] == "self"]
    native = ch.c161_binding()
    rows = []
    with TemporaryDirectory() as temp:
        work = Path(temp)
        full = ch.binding(work/"full.so", coded="v5")
        pair = ch.binding(work/"pair.so", coded="aligned")
        dll = trial.inspect_first(work)
        for src, mic, ref, _ in rates.handoff.probe.shower.streams():
            if src["player"] != "self":
                continue
            witness, = [r for r in witnesses if src == r["source"]]
            assert ch.digest(mic) == witness["mic_sha256"] and ch.digest(ref) == witness["reference_sha256"]
            routes = ch.routes(full, native, mic)
            assert routes["c161"] == witness["c161"]
            for offset in OFFSETS:
                assert not ref[:abs(offset)].any() and (offset == 0 or not ref[-abs(offset):].any())
                shifted = np.roll(ref, offset)
                base = rates.guarded(routes, full, pair, shifted)
                row = trial.compare(dll, mic, shifted, base, dict(src, reference_offset_samples=offset))
                if offset == 0:
                    assert row["old_opened"] == witness["opened"]
                if offset in BOARD_OFFSETS:
                    i = len(rows)
                    row["original_p4"] = board(work, build, 0x4652594e, 2*i, mic, shifted, row["old_opened"])
                    row["candidate_p4"] = board(work, build, 0x5752594e, 2*i+1, mic, shifted, row["opened"])
                rows.append(row)
            print(src["room"], src["mode"], src["noisy"], flush=True)
    assert len(rows) == 108
    ch.save(REPORT, dict(records=rows, p4_offsets=BOARD_OFFSETS, offsets_samples=OFFSETS,
        patch_sha256=prior["sources"]["first_reference_wait.patch"], listener_sha256=ch.sha(fw),
        parent_sha256=ch.sha(history), original_stereo_sha256=ch.sha(rates.DRIFT_BASELINE),
        script_sha256=ch.sha(Path(__file__)), policy="Fixed reference delay on unchanged room microphones. "
        "Raw or VoIP 12/48 kbit/s cue, quiet or original noise, two reused rooms. Both firmware modes "
        "cold-booted at 150 and 192 ms lag for every source. Signed offsets in samples at 16 kHz."))


if __name__ == "__main__":
    main()
