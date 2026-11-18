"""Replay the fixed v5 first-command fixtures with P4 AEC/AGC and score their PCM on CUDA."""
import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.signal import correlate
import soundfile as sf
import yaml

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import QEMU, WORK, boot, transcripts

BUILD = WORK / "frontend"
LOG = WORK / "frontend-uart.log"
STEM = "4090-c161-first-command-frontend-v5"
AUDIT = RESULTS / "4090-c161-coded-cue-v5-audit.json"


def replay(log):
    boot(BUILD, log, WORK / "frontend-flash.bin", b"FRONTEND PASS")


def parse_uart(log):
    traces = {}
    current = None
    for line in log.read_text().splitlines():
        if line.startswith("BEGIN "):
            _, channel, length, chunk, prime = line.split()
            assert current is None and channel not in traces
            current = channel
            length, chunk, prime = map(int, (length, chunk, prime))
            first = -prime % 160
            if first:
                first -= 160
            blocks = {"aec": bytearray(), "agc": bytearray()}
            events = []
        elif line.startswith("PCM "):
            _, stage, at, hex_pcm = line.split()
            assert current is not None and int(at) == first + len(blocks[stage]) // 2
            raw = bytes.fromhex(hex_pcm)
            assert len(raw) == 320
            blocks[stage].extend(raw)
        elif line.startswith("EVENT "):
            _, channel, at = line.split()
            assert channel == current
            events.append(int(at))
        elif line.startswith("END "):
            _, channel, end = line.split()
            end = int(end)
            assert channel == current and end >= length + 2048 - 160 and prime % chunk == 0
            assert all(len(buf) == 2*(end-first) for buf in blocks.values())
            traces[channel] = (length, chunk, prime, events,
                               {k: np.frombuffer(v, dtype="<i2")[-first:] for k, v in blocks.items()})
            current = None
    assert current is None and set(traces) == {"digital", "near", "far"}
    assert "FRONTEND PASS" in log.read_text()
    return traces


def command_delay(source, stage, onset):
    command = source[onset:onset+24000].astype(np.float64)
    observed = stage[onset:onset+len(command)+2048].astype(np.float64)
    energy = np.r_[0, np.cumsum(observed**2)]
    windows = energy[len(command):] - energy[:-len(command)]
    dots = correlate(observed, command, mode="valid", method="fft")
    correlations = dots / np.sqrt(np.maximum(1, windows * (command @ command)))
    lag = int(np.argmax(correlations))
    assert 0 < lag < 2048
    return lag, float(correlations[lag])


def score():
    from faster_whisper import WhisperModel

    native_log = WORK / "frontend-native-uart.log"
    assert b"BEGIN digital " in native_log.read_bytes() and b"0x026fc02b" in native_log.read_bytes()
    traces = parse_uart(LOG)
    repeat_log = WORK / "frontend-repeat-uart.log"
    repeats = parse_uart(repeat_log)
    model = Path(".local/wakeword/whisper-small-en")
    identity = read(RESULTS / "4090-parallel-melo-context-asr-plan-v1.json")["files"]
    model_hashes = {str(p): sha(p) for p in model.iterdir() if p.is_file()}
    assert model_hashes == {k: v for k, v in identity.items() if Path(k).parent == model}
    reader = WhisperModel(str(model), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    rows = []
    for witness in read(AUDIT)["witnesses"]:
        channel = witness["channel"]
        length, chunk, prime, events, stages = traces[channel]
        assert repeats[channel][:4] == traces[channel][:4]
        source_file = RESULTS / witness["wav"]
        assert sha(source_file) == witness["sha256"]
        source, rate = sf.read(source_file, dtype="int16")
        assert rate == 16000 and source.shape == (length,)
        onset = witness["command_start"]
        reference_event = witness["notification"][0]["t"]
        row = dict(channel=channel, wav_sha256=witness["sha256"], source_samples=length,
                   aec_chunk_samples=chunk, prime_samples=prime, output_samples=len(stages["agc"]),
                   input_command_onset=onset, input_notification=reference_event, agc_notifications=events,
                   source_transcript=transcripts(reader, source[onset:]), stages={})
        for stage, pcm in stages.items():
            assert np.array_equal(pcm, repeats[channel][4][stage])
            delay, corr = command_delay(source, pcm, onset)
            window = pcm[onset+delay:length+delay]
            assert len(window) == length - onset
            row["stages"][stage] = dict(pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                       command_delay_samples=delay, command_correlation=corr,
                                       command_transcript=transcripts(reader, window))
        rows.append(row)
    project = read(BUILD / "project_description.json")
    roots = dict(zip(project["build_components"], map(Path, project["build_component_paths"]), strict=True))
    fft = roots["espressif__dl_fft"]
    sr = Path(next(line.split("=", 1)[1] for line in (BUILD / "CMakeCache.txt").read_text().splitlines()
                   if line.startswith("NYX_SR_COMPONENT:")))
    lock = yaml.safe_load(Path("dependencies.lock").read_text())["dependencies"]
    dependencies = {}
    for name, directory, files in [
        ("espressif/esp-sr", sr, ["lib/esp32p4/libesp_audio_processor.a", "lib/esp32p4/libdl_lib.a"]),
        ("espressif/dl_fft", fft, ["base/dl_fft2r_fc32_ansi.c", "base/isa/esp32p4/dl_fft2r_fc32_arp4.S"]),
    ]:
        dep = lock[name]
        assert (directory / ".component_hash").read_text().strip() == dep["component_hash"]
        expected = {r["path"]: r["hash"] for r in read(directory / "CHECKSUMS.json")["files"]}
        checksums = {f: sha(directory / f) for f in files}
        assert all(expected[f] == digest for f, digest in checksums.items())
        dependencies[name] = dict(version=dep["version"], component_hash=dep["component_hash"], files=checksums)

    archive = RESULTS / f"{STEM}-uart.zip"
    with ZipFile(archive, "x", ZIP_DEFLATED) as out:
        out.write(LOG, "frontend-uart.log")
        out.write(native_log, "native-fft-trap.log")
    save(RESULTS / f"{STEM}.json", dict(
        mode="ESP32-P4 QEMU, registered external RAM, zero reference, two dl_fft ANSI kernels",
        source_report_sha256=sha(AUDIT), dependencies=dependencies,
        elf_sha256=sha(BUILD / "nyx_first_command.elf"), qemu_sha256=sha(QEMU),
        native_fft_elf_sha256=sha(WORK / "frontend-native/nyx_first_command.elf"),
        native_fft_failure="dl_fft2r_fc32_arp4_: esp.lp.setup, machine instruction 0x026fc02b",
        uart_sha256=sha(LOG), repeat_uart_sha256=sha(repeat_log), uart_archive_sha256=sha(archive),
        repeat_check="all post-prime output samples and notifications equal in two QEMU boots",
        asr=dict(model="small.en", device="cuda", compute_type="float16", files=model_hashes), records=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", nargs="+", choices=["native", "boot", "repeat", "score"])
    for action in parser.parse_args().actions:
        if action == "score":
            score()
        elif action == "native":
            boot(WORK / "frontend-native", WORK / "frontend-native-uart.log",
                 WORK / "frontend-native-flash.bin", b"0x026fc02b  ")
        else:
            replay(LOG if action == "boot" else WORK / "frontend-repeat-uart.log")
