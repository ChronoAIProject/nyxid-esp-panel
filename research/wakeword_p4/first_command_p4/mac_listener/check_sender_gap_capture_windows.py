"""Locate the handed-off dry command in the baseline and spaced full Opus uplinks."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import check_sender_command_gap as gap
import measure_command_capture as windows
import measure_connected_command_clock as clock

ch = gap.ch
OUT = gap.OUT.with_name("4090-c161-sender-gap-capture-windows.json")


def measure(logs, board, command, decoder):
    if not board["opened"]:
        return dict(status="no_open")
    capture = board["uplink"]
    index = board["index"]
    uart = logs.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
    uplink = gap.tx.panel.grid.gap.uplink
    packets = uplink.packets(uart, index, capture["samples"], 0)
    audio = uplink.decode(decoder, packets)
    assert len(packets) == capture["packets"]
    assert ch.digest(audio) == capture["pcm_sha256"] == board["qwen"]["pcm_sha256"]
    measurement = windows.locate(command, audio)
    reading = dict(measurement, pcm_sha256=capture["pcm_sha256"], qwen_complete=board["qwen"]["complete"],
                   turbo_complete=board["turbo"], opened=board["opened"], packets=len(packets))
    if measurement["status"] == "three_windows_matched":
        # Remove the located 160 ms opening window, with a 20 ms guard at both ends.
        # Preserve the rest of the decoded recording and its clock for a same-trial control.
        start = measurement["anchors"]["begin"]["captured_sample"]
        muted = audio.copy()
        stop = min(len(muted), start + windows.WIDTH * windows.DECIMATION + 320)
        muted[max(0, start-320):stop] = 0
        control = windows.locate(command, muted)
        reading.update(muted_samples=[max(0, start-320), stop], muted_status=control["status"])
    return reading


def main():
    paths = {"first": gap.study.timing.GUEST, "second": gap.study.GUEST, "spaced": gap.OUT}
    reports = {name: ch.read(path) for name, path in paths.items()}
    sources = [Path(__file__), Path(windows.__file__), Path(clock.__file__)]
    sources.extend(p for p, _, _ in clock.PACKS.values())
    decoder = gap.tx.panel.grid.gap.uplink.decoder_api()
    counts = defaultdict(Counter)
    rows = []
    with ExitStack() as stack:
        logs = {}
        for name, path in paths.items():
            archive = path.with_suffix(".zip")
            assert ch.sha(archive) == reports[name]["archive_sha256"]
            logs[name] = stack.enter_context(ZipFile(archive))
            sources.extend((archive, path))
        originals = {batch: {row["index"]: row for row in reports[batch]["records"]} for batch in ("first", "second")}
        tracks = {}
        for paired in reports["spaced"]["records"]:
            batch, new = paired["batch"], paired["board"]
            old = originals[batch][paired["previous_case"]]
            s = new["source"]
            assert old["source"] == dict(s, start=s["start"]+reports["spaced"]["gap_samples"])
            key = batch, s["index"]
            if key not in tracks:
                pack = "balanced" if batch == "first" else "additional"
                dry, boundary, _ = clock.track(s["index"], pack)
                assert boundary == s["command_at"]
                tracks[key] = dry[boundary:]
            command = tracks[key]
            result = dict(batch=batch, track=s["index"], external_db=s["external_db"],
                          command=s["command"], command_pcm_sha256=ch.digest(command))
            for variant, board, archive in (("original", old, logs[batch]), ("spaced", new, logs["spaced"])):
                detection = measure(archive, board, command, decoder)
                matched = detection["status"] == "three_windows_matched"
                group = counts[f"{batch}.{variant}"]
                group.update(trials=1, three_windows=int(matched))
                group[detection["status"]] += 1
                if detection["status"] != "no_open":
                    miss = not detection["qwen_complete"]
                    group.update(qwen_miss=int(miss), qwen_miss_three_windows=int(miss and matched))
                if matched:
                    group["three_windows_after_mute"] += int(detection["muted_status"] == "three_windows_matched")
                result[variant] = detection
            rows.append(result)
    assert len(rows) == 72 and len(tracks) == 24
    with TemporaryDirectory(dir=OUT.parent) as folder:
        p = Path(folder)/OUT.name
        ch.save(p, dict(sources={path.name: ch.sha(path) for path in sources}, summary=counts, records=rows,
                        policy="Compare three 160 ms waveform windows of the known dry command with the full "
                        "decoded P4 uplink using the existing Mac capture reader. If all three match, mute "
                        "the first aligned window with a 20 ms guard at both ends and compare again. "
                        "A match locates waveform segments and does not establish word intelligibility."))
        p.replace(OUT)


if __name__ == "__main__":
    main()
