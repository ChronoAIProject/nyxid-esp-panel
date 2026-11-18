"""Feed the archived P4 AGC samples through the P4 capture FIFO and Opus encoder."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, sha
from run_c161_first_command_v5 import WORK, boot, score
from run_c161_frontend_v5 import STEM as FRONTEND, parse_uart

BUILD = WORK / "after-agc-build"
FIXTURES = WORK / "after-agc-inputs"
LOG = WORK / "after-agc-uart.log"
STEM = "4090-c161-first-command-after-agc-opus-v5"


def prepare():
    report = RESULTS / f"{FRONTEND}.json"
    archive = RESULTS / f"{FRONTEND}-uart.zip"
    frontend = read(report)
    assert sha(archive) == frontend["uart_archive_sha256"]
    with TemporaryDirectory() as tmp, ZipFile(archive) as z:
        log = Path(tmp) / "uart.log"
        log.write_bytes(z.read("frontend-uart.log"))
        assert sha(log) == frontend["uart_sha256"]
        traces = parse_uart(log)
    rows = []
    events = []
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for channel in ("digital", "near", "far"):
        rec = next(r for r in frontend["records"] if r["channel"] == channel)
        length, chunk, prime, notices, stages = traces[channel]
        assert len(notices) == 1 and notices == rec["agc_notifications"]
        pcm = stages["agc"]
        stage = rec["stages"]["agc"]
        assert hashlib.sha256(pcm.tobytes()).hexdigest() == stage["pcm_sha256"]
        # Keep the measured DSP delay at the end, so every command has the original trailing context.
        delay = stage["command_delay_samples"]
        audio = pcm[:length+delay]
        assert len(audio) == length + delay
        wav = FIXTURES / f"4090-c161-coded-cue-v5-command-{channel}.wav"
        sf.write(wav, audio, 16000, subtype="PCM_16")
        rows.append(dict(channel=channel, wav=wav.name, sha256=sha(wav),
                         command_start=rec["input_command_onset"]+delay, notification=[dict(t=notices[0])],
                         aec_chunk_samples=chunk, prime_samples=prime, frontend_pcm_sha256=stage["pcm_sha256"]))
        events.append(str(notices[0]))
    assert len({(r["aec_chunk_samples"], r["prime_samples"]) for r in rows}) == 1
    (FIXTURES / "agc_replay.h").write_text(f"constexpr int capture_step = {chunk}, prime = {prime};\n"
                                          f"constexpr int events[] = {{{', '.join(events)}}};\n")
    (FIXTURES / "fixtures.json").write_text(json.dumps(dict(frontend_report_sha256=sha(report), witnesses=rows), indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", nargs="+", choices=["prepare", "boot", "score"])
    for action in parser.parse_args().actions:
        if action == "prepare":
            prepare()
        elif action == "boot":
            boot(BUILD, LOG, WORK / "after-agc-flash.bin")
        else:
            score(BUILD, LOG, STEM, FIXTURES / "fixtures.json", FIXTURES, clocked=True)
