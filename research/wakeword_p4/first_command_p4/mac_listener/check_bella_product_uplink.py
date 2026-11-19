"""Compare product AEC/AGC and delayed raw FIFO PCM on the same fixed-tone wake clocks."""
from hashlib import sha256
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


import replay
from transition import CODEC, packets, samples
from diagnose_panel_command_levels import reader
from check_balanced_connected_commands import evaluate
from check_panel_qwen_asr import load
from check_c161_command_playback_level import recognition
from run_c161_first_command_v5 import decode, decoder_api

BASE = replay.RESULTS / "4090-c161-bella-command-prefix.json"
WATCHDOG = replay.RESULTS / "4090-c161-bella-uart-watchdog.json"
OUT = replay.RESULTS / "4090-c161-bella-product-uplink.json"
RAW = Path(".local/wakeword/c161-panel-middle-uart") / WATCHDOG.with_suffix(".zip").name
WORK = Path(".local/wakeword/c161-bella-product-uplink")


def capture(work, archive, baseline_logs, trace, row, checked, gpu):
    index = row["index"]
    old = row["uplink"]
    count = old["samples"]
    assert checked["index"] == index and checked["opened"] == row["opened"]
    assert sha256(trace).hexdigest() == checked["paced"]["uart_sha256"]
    text = trace.decode().replace("\r\n", "\n")
    emitted, first, length, *_ = replay.values(text, "FINISH")
    end, produced, start = replay.values(text, "OPEN")
    assert [end] == row["opened"] and produced == end//160*160
    assert length == count == emitted-first and first == start == produced-32000
    assert old["delayed_raw_samples"] == count and old["processed_samples"] == 0
    assert row["reference_sha256"] == sha256(bytes(2*replay.values(text, "READY")[1])).hexdigest()

    # With zero R and no TX the earliest raw C161/cue event determines the first read.
    hits = [int(t) for line in text.splitlines() if line.startswith("HITS ")
            for t in line.split()[2:] if int(t) > 0]
    assert (min(hits)+255)//256*256 == end < row["source"]["command_at"]
    control = samples(text, "AUDIO", count)
    processed = samples(text, "PROCESSED", count)
    assert replay.digest(control) == old["fifo_sha256"] == checked["paced"]["pcm_sha256"]["AUDIO"]
    assert replay.digest(processed) == checked["paced"]["pcm_sha256"]["PROCESSED"]
    payload = struct.pack("<3I", 0x3552504f, index, count) + processed.tobytes() + control.tobytes()
    codec_log = work / "opus.log"
    replay.boot(CODEC, codec_log, work/"flash.bin", b"UPLINK PASS", (2<<20, payload))
    uart = codec_log.read_text()
    assert "Task watchdog got triggered" not in uart
    actual = packets(uart, index, count, 0)
    comparison = packets(uart, index, count, 1)
    original = baseline_logs.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
    assert comparison == packets(original, index, count, 0)
    proc, model, turbo, decoder = gpu
    assert len(actual) == len(comparison) == old["packets"]
    assert replay.digest(decode(decoder, comparison)) == old["pcm_sha256"]
    full = decode(decoder, actual)
    expected = row["source"]["command"]
    archive.write(codec_log, f"{index:02}-opus.log")
    return dict(index=index, file=checked["file"], db=row["source"]["external_db"], command=expected,
                opened=end, command_at=row["source"]["command_at"], first=first, pcm_samples=count,
                packets=len(actual), processed_fifo_sha256=replay.digest(processed),
                raw_fifo_sha256=replay.digest(control), capture_log_sha256=checked["paced"]["uart_sha256"],
                opus_log_sha256=replay.sha(codec_log), processed_decoded_sha256=replay.digest(full),
                raw_decoded_sha256=old["pcm_sha256"], raw_qwen=row["qwen"], raw_turbo=old["turbo"],
                processed_qwen=evaluate(proc, model, full, expected),
                processed_turbo=recognition(turbo, full, expected))


def main():
    baseline, watchdog = replay.read(BASE), replay.read(WATCHDOG)
    assert watchdog["baseline_sha256"] == replay.sha(BASE) and replay.sha(RAW) == watchdog["raw_sha256"]
    assert replay.sha(BASE.with_suffix(".zip")) == baseline["archive_sha256"]
    assert replay.sha(CODEC/"nyx_first_command.bin") == baseline["codec"]["opus_sha256"]
    assert replay.sha(replay.QEMU) == baseline["codec"]["qemu_sha256"]
    proc, model, qwen = load()
    assert qwen == baseline["qwen"]
    gpu = proc, model, reader(baseline["turbo"]), decoder_api()
    WORK.mkdir(parents=True, exist_ok=True)
    archive = OUT.with_suffix(".zip")
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(RAW) as source, ZipFile(BASE.with_suffix(".zip")) as old:
        work = Path(tmp)
        records = []
        with ZipFile(work/archive.name, "w", ZIP_DEFLATED) as logs:
            for row, checked in zip(baseline["captures"], watchdog["records"], strict=True):
                trace = source.read(f"paced/{row['index']:03}.log")
                records.append(capture(work, logs, old, trace, row, checked, gpu))
                print(len(records), "of 24", flush=True)
        assert len(records) == len({r["file"] for r in records}) == 24
        replay.save(work/OUT.name, dict(records=records, qwen=qwen, turbo=baseline["turbo"],
                    codec=baseline["codec"], model=baseline["model"], capture_firmware=watchdog["firmware"]["paced"],
                    board_base=replay.MAC, archive_sha256=replay.sha(work/archive.name),
                    sources={p.name:replay.sha(p) for p in (BASE, WATCHDOG, RAW, Path(__file__))},
                    policy="Original 24 paired af_bella rear development mixes. Identical zero R, wake clock, "
                           "two-second preroll, FIFO end and frame count. Each lane has a fresh 24 kb/s P4 Opus "
                           "encoder and decoder. Compare the full simulated ESP AEC/AGC lane against the "
                           "previously selected 90 ms delayed raw microphone. Ignore network/setup fill packets."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
