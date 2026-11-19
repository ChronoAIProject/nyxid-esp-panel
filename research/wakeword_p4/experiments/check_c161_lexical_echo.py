"""Replay ordinary connected requests with and without a time-aligned speaker reference."""
from pathlib import Path
import struct

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

from check_c161_v5_playback_reference import BOT, BUILD, ROOMS, RATE
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, sha
from run_c161_first_command_v5 import WORK, QEMU, boot
from select_c161_synthesized_cue_v2 import load_wav

OUT = WORK / "lexical-echo"
STEM = "4090-c161-lexical-echo"
FORWARD = RESULTS / "4090-c161-signed-forward-v1.json"
FIRMWARE = RESULTS / "4090-c161-v5-playback-reference-guard.json"
AT = 5 * RATE
LENGTH = 10 * RATE


def cases():
    forward = read(FORWARD)["records"]
    bot_file = RESULTS / f"{BOT}-pcm-v1.npz"
    report = read(RESULTS / f"{BOT}-v1.json")
    assert sha(bot_file) == report["pcm_archive_sha256"]
    rooms = read(ROOMS)
    with np.load(bot_file) as bank:
        for bot_key, ext_id in (("af_kore_digital", 0), ("bm_fable_digital", 12)):
            row = next(r for r in report["records"] if r["key"] == bot_key)
            bot = bank[bot_key]
            assert digest(bot) == row["pcm_sha256"]
            demonstration = row["spans"][-1]
            self_clip = bot[demonstration["start"]:demonstration["end"]]
            src = forward[ext_id]
            request = load_wav(src)[src["start"]:src["end"]]
            plain = bot[:LENGTH].astype(float)
            # The same prefix trains the echo canceller before the change in playback.
            saying = plain.copy()
            saying[AT:AT+len(self_clip)] = self_clip
            near = np.zeros(LENGTH)
            near[AT:AT+len(request)] = request
            for scene in (0, 1):
                room = next(r["room"] for r in rooms["records"] if r["scene"] == scene)
                path = Path(".local/wakeword/mit-rirs") / room
                assert sha(path) == rooms["files"][str(path)]
                impulse, rate = sf.read(path)
                assert rate == RATE
                impulse /= np.linalg.norm(impulse)
                own_echo = fftconvolve(saying, impulse)[:LENGTH]
                chat_echo = fftconvolve(plain, impulse)[:LENGTH]
                # Equal target/ordinary playback energy over the external utterance.
                period = slice(AT, AT+len(request))
                gain = np.linalg.norm(near[period]) / np.linalg.norm(chat_echo[period])
                own_echo *= gain
                chat_echo *= gain
                silent = np.zeros(LENGTH)
                tracks = {
                    "self": (own_echo, saying),
                    "same_mic_external": (own_echo, silent),
                    "external_only": (near, silent),
                    "external_over_reply": (near+chat_echo, plain),
                    "simultaneous_mentions": (near+own_echo, saying),
                    "reply_only": (chat_echo, plain),
                }
                scale = min(1., 28000/max(np.abs(v).max() for pair in tracks.values() for v in pair))
                for role, (mic, ref) in tracks.items():
                    mic, ref = (np.rint(scale*v).astype("<i2") for v in (mic, ref))
                    key = f"{bot_key}-{scene}-{role}"
                    yield key, mic, ref, dict(role=role, bot=bot_key, external_voice=src["voice"], room=room,
                          external_id=ext_id, echo_gain=gain, common_gain=scale, mic_sha256=digest(mic),
                          ref_sha256=digest(ref), start=AT, external_end=AT+len(request),
                          self_end=AT+len(self_clip), external_command=src["carrier"],
                          external_first_phone=AT+src["first_phone"]-src["start"])


def replay():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read(FIRMWARE)
    assert sha(BUILD/"nyx_first_command.elf") == prior["elf_sha256"] and sha(QEMU) == prior["qemu_sha256"]
    for key, mic, ref, meta in cases():
        payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
        log = OUT / f"{key}.log"
        boot(BUILD, log, OUT/"flash.bin", b"FRONTEND PASS", (2 << 20, payload))
        print(key, sha(log), flush=True)


if __name__ == "__main__":
    replay()
