"""Stress the standalone P4 detector across rearm boundaries on preloaded digital M/R."""
import argparse
from pathlib import Path
import re
import struct
from zipfile import ZipFile, ZIP_DEFLATED

import check_fixed_sound_runtime as profile

LABELS = ["six-cups", "quiet/erased", "command1-talk1-music1", "reference/self-restored-R", "quiet/nyx"]
RESULT = profile.RESULTS / "4090-fixed-sound-rearm.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--idf-python", required=True, type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    history = profile.room.read(profile.REPORT)
    prior = {r["label"]: r for r in history["records"]}
    source = profile.HERE / "rearm_fixed_sound.c"
    flash = profile.build(args.board.resolve(), work, True, args.idf_python.absolute(), source)
    clips = {}
    for label, mic, ref, _, _, _ in profile.inputs():
        if label in LABELS:
            data = profile.run_guest.payload(mic, ref)
            assert profile.digest(data) == prior[label]["serialized_input_sha256"]
            clips[label] = data
    sizes = [struct.unpack_from("<I", clips[label])[0] for label in LABELS]
    data = struct.pack("<5I", *sizes) + b"".join(clips[label][4:] for label in LABELS)
    assert len(data) <= 2 << 20 and profile.digest(profile.QEMU.read_bytes()) == history["qemu_sha256"]
    flash[2<<20:(2<<20)+len(data)] = data
    image = work / "flash.bin"
    image.write_bytes(flash)
    log = work / "rearm.log"
    if log.exists():
        log.rename(work / ("previous-" + profile.digest(log.read_bytes()) + ".log"))
    profile.run_guest.boot(profile.QEMU, image, log, timeout=240)
    uart = log.read_text()
    assert uart.count("REARM resident") == uart.count("BOARD PASS") == 1
    decisions = re.findall(r"^REARM decision (\d+) (\d+) (\d+) (\d+)$", uart, re.M)
    assert len(decisions) == 64
    rows = []
    for i, (epoch, clip, cue, clock) in enumerate(decisions):
        negative = i % (len(LABELS) - 1)
        assert int(epoch) == i
        hits = {stage: [int(a), int(b)] for stage, a, b in re.findall(
            rf"^REARM heap {i} (\w+) (\d+) (\d+)$", uart, re.M)}
        assert set(hits) == {"before", "ready", "released"}
        rows.append(dict(epoch=i, negative=LABELS[negative], hit=LABELS[int(clip)] if int(cue) else None,
                         cue_sample=int(cue), read_end=int(clock), negative_samples=sizes[negative], heap=hits))
    target = prior[LABELS[-1]]
    expected = target["fixed"]["epochs"][0]["events"]
    timely = sum(r["hit"] == LABELS[-1] and r["read_end"] - r["negative_samples"] < target["command_at"] for r in rows)
    same = sum(r["hit"] == LABELS[-1] and r["cue_sample"] - r["negative_samples"] == expected["C"][0]
               and r["read_end"] - r["negative_samples"] == expected["O"][0] for r in rows)
    summary = dict(cycles=len(rows), timely=timely, matched_isolated_clock=same,
                   early_negative_hits=sum(r["hit"] == r["negative"] for r in rows),
                   heap_restored=sum(r["heap"]["released"] == r["heap"]["before"] for r in rows),
                   input_samples=sum(r["read_end"] for r in rows),
                   watchdog_messages=uart.count("Task watchdog got triggered") + uart.count("Interrupt wdt timeout"),
                   panics=uart.count("Guru Meditation"))
    archive = RESULT.with_suffix(".zip")
    with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as z:
        z.write(log, "rearm.log")
    report = dict(summary=summary, records=rows, prior_sha256=profile.digest(profile.REPORT.read_bytes()),
                  sources={label: prior[label]["serialized_input_sha256"] for label in LABELS},
                  qemu_sha256=history["qemu_sha256"], guest_sha256=profile.digest((work/"board.bin").read_bytes()),
                  guest_elf_sha256=profile.digest((work/"board.elf").read_bytes()), payload_sha256=profile.digest(data),
                  source_sha256=profile.digest(source.read_bytes()), uart_sha256=profile.digest(log.read_bytes()),
                  archive_sha256=profile.digest((work/archive.name).read_bytes()),
                  scope="One P4 QEMU boot. Four consumed negative clips preceding the same quiet 077 positive. "
                        "Synchronous microphone/reference PCM is loaded into PSRAM before the first epoch. "
                        "The isolated detector releases state after a request. Audio devices and network calls "
                        "are outside the fixture. Repetitions add zero independent false-alarm exposure.")
    profile.room.save(work/RESULT.name, report)
    (work/archive.name).replace(archive)
    (work/RESULT.name).replace(RESULT)
    print(summary)


if __name__ == "__main__":
    main()
