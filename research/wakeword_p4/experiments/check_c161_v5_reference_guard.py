"""Validate a causal reference veto in QEMU on fixed and overlapping playback tracks."""
import io
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_v5_playback_reference import AUDIT, BOT, BUILD, OUT, RATE, ROOMS, STEM, cases
from check_c161_v3_two_anchor_v1 import binding
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from generate_c161_coded_cue_v5 import WAV
from replay_c161_pcm_cues_v1 import detect
from run_c161_first_command_v5 import boot, QEMU
from run_c161_frontend_v5 import parse_uart

PREFIX = f"{STEM}-guard"
GUARD = RESULTS.parent / "portable/c161_coded_cue_reference_v5.h"
WINDOW = 1440 + 8 * 256


def overlap_cases():
    assert sha(WAV) == read(AUDIT)["cue_sha256"]
    cue, rate = sf.read(WAV, dtype="int16")
    assert rate == RATE
    bot_path = RESULTS / f"{BOT}-pcm-v1.npz"
    assert sha(bot_path) == read(RESULTS / f"{BOT}-v1.json")["pcm_archive_sha256"]
    with np.load(bot_path) as bank:
        bot = bank["bm_fable_digital"][:13 * RATE].astype(float)
    command_archive = RESULTS / "4090-c161-v5-household-rooms-dry.zip"
    assert sha(command_archive) == read(ROOMS)["waveform_archive_sha256"]
    with ZipFile(command_archive) as z:
        command, rate = sf.read(io.BytesIO(z.read("bm_lewis-1-command.wav")), dtype="int16")
    assert rate == RATE
    utterance = np.r_[cue, np.zeros(320, np.int16), command]
    start = 5 * RATE
    ref = bot.copy()
    ref[start:start + len(utterance)] = utterance
    rir_manifest = read(RESULTS / "4090-mit-rir-audit.json")["records"]
    for prefix in ("h003_", "h010_"):
        rir, = [r for r in rir_manifest if Path(r["path"]).name.startswith(prefix) and r["split"] == "training"]
        path = Path(".local/wakeword/mit-rirs") / rir["path"]
        assert sha(path) == rir["sha256"]
        h, rate = sf.read(path)
        assert rate == RATE
        echo = .45 * fftconvolve(ref, h / np.linalg.norm(h))[:len(ref)]
        for gap in (0, 3 * RATE):
            external = np.zeros(len(ref))
            at = start + gap
            external[at:at + len(utterance)] = utterance
            mic = external + echo
            gain = min(1., 28000 / max(abs(mic).max(), abs(ref).max()))
            mic, reference = (np.rint(gain * v).astype("<i2") for v in (mic, ref))
            key = f"{prefix[:4]}-gap-{gap}"
            yield key, mic, reference, dict(role="overlap", room=rir["path"], rir_sha256=rir["sha256"],
                                            self_start=start, external_start=at, common_gain=gain,
                                            self_reference_gain=1., echo_gain=.45, expected_command="Play the evening news.",
                                            command_start=at + len(cue) + 320, command_samples=len(command))


def main():
    previous = RESULTS / f"{STEM}.json"
    controls = {r["key"]: r for r in read(previous)["records"]}
    root = OUT.parent / "echo-guard"
    root.mkdir(exist_ok=True)
    records = []
    archive = RESULTS / f"{PREFIX}-uart.zip"
    with TemporaryDirectory() as temp, ZipFile(archive, "x", ZIP_DEFLATED) as z:
        lib = binding(Path(temp) / "v5.so", coded="v5")
        for key, mic, ref, meta in (*cases(), *overlap_cases()):
            log = root / f"{key}.log"
            payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
            boot(BUILD, log, root / "flash.bin", b"FRONTEND PASS", (2 << 20, payload))
            length, chunk, prime, events, stages = parse_uart(log, ("echo",))["echo"]
            assert length == len(mic) and chunk == 256 and prime == 81664
            if key in controls:
                old = controls.pop(key)
                assert (digest(mic), digest(ref)) == (old["mic_sha256"], old["reference_sha256"])
                assert events == old["agc_events"] and all(digest(w) == old["stages"][s]["pcm_sha256"] for s, w in stages.items())
            refs = [e["t"] for e in detect(lib, [ref])]
            allowed = [int(line.split()[1]) for line in log.read_text().splitlines() if line.startswith("ALLOWED ")]
            assert allowed == [t for t in events if not any(0 <= t - r <= WINDOW for r in refs)]
            z.write(log, log.name)
            records.append(dict(key=key, source=meta, mic_sha256=digest(mic), reference_sha256=digest(ref),
                                uart_sha256=sha(log), reference_events=refs, agc_events=events, allowed=allowed))
            print(key, refs, events, allowed, flush=True)
    assert not controls and len(records) == 16
    save(RESULTS / f"{PREFIX}.json", dict(previous_sha256=sha(previous), guard_sha256=sha(GUARD),
         elf_sha256=sha(BUILD / "nyx_first_command.elf"), qemu_sha256=sha(QEMU), window_samples=WINDOW,
         uart_archive_sha256=sha(archive), records=records))


if __name__ == "__main__":
    main()
