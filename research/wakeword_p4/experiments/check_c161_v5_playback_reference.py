"""Replay two synchronous capture channels through P4 ESP-SR and the fixed v5 detector."""
import argparse
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_v3_two_anchor_v1 import binding
from check_c161_v5_command_ns import digest
from check_c161_v5_far_command_noise import normalized
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import WORK, QEMU, boot, transcripts
from run_c161_frontend_v5 import parse_uart

BUILD = WORK / "echo-build"
OUT = WORK / "echo-records"
STEM = "4090-c161-v5-playback-reference"
BOT = "4090-c161-bot-replies"
AUDIT = RESULTS / "4090-c161-coded-cue-v5-audit.json"
ROOMS = RESULTS / "4090-c161-v5-household-rooms-plan.json"
PRIOR = RESULTS / "4090-c161-first-command-frontend-v5.json"
RATE = 16000


def cases():
    witness = next(r for r in read(AUDIT)["witnesses"] if r["channel"] == "digital")
    source = RESULTS / witness["wav"]
    assert sha(source) == witness["sha256"]
    digital, rate = sf.read(source, dtype="int16")
    start, onset = witness["cue_start"], witness["command_start"]
    assert rate == RATE
    external = digital.astype(float)
    external[:start] = 0
    bot_path = RESULTS / f"{BOT}-pcm-v1.npz"
    bot_report = read(RESULTS / f"{BOT}-v1.json")
    assert sha(bot_path) == bot_report["pcm_archive_sha256"]
    bot_record = next(r for r in bot_report["records"] if r["key"] == "af_kore_digital")
    with np.load(bot_path) as z:
        bot = z[bot_record["key"]]
        assert digest(bot) == bot_record["pcm_sha256"]
        bot = bot[:len(digital)].astype(float)
    self_play = bot.copy()
    self_play[start:] = external[start:]
    rooms = read(ROOMS)
    for scene in (0, 1):
        room = next(row["room"] for row in rooms["records"] if row["scene"] == scene)
        path = Path(".local/wakeword/mit-rirs") / room
        assert sha(path) == rooms["files"][str(path)]
        impulse, rate = sf.read(path)
        assert rate == RATE
        impulse /= np.linalg.norm(impulse)
        echo = fftconvolve(bot, impulse)[:len(bot)]
        own_echo = fftconvolve(self_play, impulse)[:len(bot)]
        # Match the command and concurrent assistant on the active external command frames.
        command = external[onset:]
        powers = np.mean(np.pad(command**2, (0, -len(command) % 320)).reshape(-1, 320), axis=1)
        active = np.repeat(powers >= powers.max() * .001, 320)[:len(command)]
        echo_gain = np.linalg.norm(command[active]) / np.linalg.norm(echo[onset:][active])
        waves = {"external": (external + echo_gain * echo, bot),
                 "self": (echo_gain * own_echo, self_play),
                 "bot_only": (echo_gain * echo, bot)}
        gain = min(1., 28000 / max(np.max(np.abs(v)) for pair in waves.values() for v in pair))
        for role, (signal, reference) in waves.items():
            mic = np.rint(gain * signal).astype("<i2")
            reference = np.rint(gain * reference).astype("<i2")
            for setting, ref in (("zero", np.zeros_like(reference)), ("speaker", reference)):
                key = f"{scene}-{role}-{setting}"
                yield key, mic, ref, dict(room=room, rir_sha256=sha(path), scene=scene, role=role,
                                          reference=setting, start=start, command_onset=onset,
                                          mic_sha256=digest(mic), reference_sha256=digest(ref),
                                          common_gain=gain, echo_gain=echo_gain, active_samples=int(active.sum()))


def replay():
    OUT.mkdir(parents=True, exist_ok=True)
    for key, mic, ref, _ in cases():
        payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
        boot(BUILD, OUT / f"{key}.log", OUT / "flash.bin", b"FRONTEND PASS", (2 << 20, payload))
        print(key, flush=True)


def score():
    previous = read(PRIOR)
    model = Path(".local/wakeword/whisper-small-en")
    assert all(sha(Path(p)) == h for p, h in previous["asr"]["files"].items())
    reader = WhisperModel(str(model), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    assert reader.model.device == "cuda"
    # Re-run the original three zero-reference streams on the same ELF, including every PCM sample.
    regression = WORK / "echo-regression.log"
    boot(BUILD, regression, OUT / "flash.bin", b"FRONTEND PASS")
    controls = parse_uart(regression)
    for r in previous["records"]:
        length, _, _, events, stages = controls[r["channel"]]
        assert length == r["source_samples"] and events == r["agc_notifications"]
        for stage, pcm in stages.items():
            assert digest(pcm) == r["stages"][stage]["pcm_sha256"]
    lag, = {v["command_delay_samples"] for r in previous["records"] for v in r["stages"].values()}
    records = []
    witnesses = RESULTS / f"{STEM}-stereo.zip"
    logs = RESULTS / f"{STEM}-uart.zip"
    with TemporaryDirectory() as temp, ZipFile(logs, "x", ZIP_DEFLATED) as uart, ZipFile(witnesses, "x", ZIP_DEFLATED) as audio:
        lib = binding(Path(temp) / "v5.so", coded="v5")
        for key, mic, ref, row in cases():
            log = OUT / f"{key}.log"
            length, chunk, prime, events, stages = parse_uart(log, ("echo",))["echo"]
            assert length == len(mic) and prime == 81664 and chunk == 256
            uart.write(log, f"{key}.log")
            stereo = Path(temp) / f"{key}.wav"
            sf.write(stereo, np.column_stack((mic, ref)), RATE, subtype="PCM_16")
            stereo_data, rate = sf.read(stereo, dtype="int16")
            assert rate == RATE and np.array_equal(stereo_data.T, np.stack((mic, ref)))
            audio.write(stereo, stereo.name)
            onset = row["command_onset"]
            row.update(key=key, wav_sha256=sha(stereo), samples=length, uart_sha256=sha(log),
                       input_events=[e["t"] for e in detect(lib, [mic])], agc_events=events, stages={})
            for name, wave in (("input", mic), *stages.items()):
                offset = 0 if name == "input" else lag
                command = wave[onset + offset:length + offset]
                assert len(command) == length - onset
                words = transcripts(reader, command)
                start = row["start"] + offset
                rms = float(np.sqrt(np.mean(wave[start:length + offset].astype(float)**2)))
                row["stages"][name] = dict(pcm_sha256=digest(wave), command=words, command_sha256=digest(command),
                                           exact=normalized(words["text"]) == ["lower", "the", "fan", "speed"],
                                           rms_from_cue=rms)
            records.append(row)
            print(key, row["input_events"], events, row["stages"]["agc"]["command"]["text"], flush=True)
    save(RESULTS / f"{STEM}.json", dict(source_audit_sha256=sha(AUDIT), room_plan_sha256=sha(ROOMS),
         bot_sha256=sha(RESULTS / f"{BOT}-v1.json"), prior_frontend_sha256=sha(PRIOR),
         elf_sha256=sha(BUILD/"nyx_first_command.elf"), qemu_sha256=sha(QEMU),
         fft="ESP32-P4 QEMU with component ANSI forward and inverse FFT",
         delay_for_command_crops=lag, regression_uart_sha256=sha(regression),
         uart_archive_sha256=sha(logs), stereo_archive_sha256=sha(witnesses), records=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actions", nargs="+", choices=("replay", "score"))
    for action in parser.parse_args().actions:
        {"replay": replay, "score": score}[action]()
