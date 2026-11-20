"""Record one external playback on the rev1 M/R diagnostic, or decode a saved UART dump."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import wave
import zlib

RATE = 16000
FRAMES = 8 * RATE
HEADER = re.compile(rb"MR_BEGIN epoch=(\d+) first=(\d+) frames=(\d+) end=(\w+) elf=([0-9a-f]{64})")
FOOTER = re.compile(rb"MR_END crc32=([0-9a-f]{8})")


def decode(lines, expected_elf=None):
    header = None
    checksum = None
    pcm = bytearray()
    for raw in lines:
        line = raw.strip()
        if line.startswith(b"MR_BEGIN"):
            match = HEADER.fullmatch(line)
            if not match or header:
                raise ValueError("Expected one well-formed dump per input")
            epoch, first, frames, end, elf = match.groups()
            header = dict(epoch=int(epoch), first_epoch_sample=int(first), frames=int(frames),
                          end=end.decode(), descriptor_elf_sha256=elf.decode())
            if not 0 < header["frames"] <= FRAMES or (header["end"] == "full" and header["frames"] != FRAMES):
                raise ValueError("Invalid frame count")
            if expected_elf and expected_elf.lower() != elf.decode():
                raise ValueError("UART and expected application ELF differ")
        elif line.startswith(b"MR_DATA"):
            _, offset, encoded = line.split()
            if header is None or checksum is not None or int(offset) != len(pcm):
                raise ValueError("Missing, duplicated or unordered PCM")
            chunk = base64.b64decode(encoded, validate=True)
            remaining = header["frames"] * 4 - len(pcm)
            if not chunk or len(chunk) != min(192, remaining):
                raise ValueError("Invalid UART chunk size")
            pcm.extend(chunk)
        elif line.startswith(b"MR_END"):
            match = FOOTER.fullmatch(line)
            if not match or not header or checksum is not None or len(pcm) != header["frames"] * 4:
                raise ValueError("Incomplete or duplicate capture")
            checksum = match[1].decode()
            if checksum != f"{zlib.crc32(pcm):08x}":
                raise ValueError("Device and host PCM checksums differ")
    if checksum is None:
        raise ValueError("No complete UART dump")
    return pcm, dict(**header, rate=RATE, channels=["mic", "reference"], crc32=checksum,
                     pcm_sha256=hashlib.sha256(pcm).hexdigest())


def write_wav(path, pcm):
    with path.open("xb") as output, wave.open(output, "wb") as wav:
        wav.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)


def record(args):
    import serial

    player = Path("/usr/bin/afplay")
    if not player.is_file():
        raise ValueError("Mac system audio player required")
    playback = args.playback.resolve(strict=True)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    uart_path = args.output / "uart.log"
    port = serial.Serial(port=None, baudrate=115200, timeout=.25, write_timeout=2)
    port.port = args.port
    port.dtr = port.rts = False
    with port, uart_path.open("xb", buffering=0) as log:
        pending = bytearray()

        def receive(deadline, token=None):
            while time.monotonic() < deadline:
                data = port.read(port.in_waiting or 1)
                log.write(data)
                pending.extend(data)
                while b"\n" in pending:
                    end = pending.index(b"\n") + 1
                    line = bytes(pending[:end]).strip()
                    del pending[:end]
                    if line.startswith(b"MR_ERROR"):
                        raise RuntimeError(line.decode(errors="replace"))
                    if token and (line == token or (token == b"MR_END" and FOOTER.fullmatch(line))):
                        return
            if token:
                raise TimeoutError(f"No {token.decode()} before deadline. UART saved at {uart_path}")

        port.reset_input_buffer()
        port.write(b"nyx-mr arm\r")
        receive(time.monotonic() + 8, b"MR_ARMED")
        start = time.monotonic()
        with subprocess.Popen([str(player), str(playback)]) as playing:
            try:
                receive(start + 8.4)
                playing.wait(timeout=.2)
                if playing.returncode:
                    raise RuntimeError("afplay failed")
            finally:
                if playing.poll() is None:
                    playing.terminate()
        port.write(b"nyx-mr dump\r")
        receive(time.monotonic() + 150, b"MR_END")
        with uart_path.open("rb") as recorded:
            pcm, report = decode(recorded, args.elf)
        write_wav(args.output / "mr.wav", pcm)
        report.update(playback=playback.name, playback_sha256=hashlib.sha256(playback.read_bytes()).hexdigest())
        (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
        port.write(b"nyx-mr clear\r")
        receive(time.monotonic() + 5, b"MR_CLEARED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    live = modes.add_parser("record", help="Play a short WAV and save M/R, UART and identity in a new directory")
    live.add_argument("playback", type=Path)
    live.add_argument("output", type=Path)
    live.add_argument("--port", required=True)
    saved = modes.add_parser("decode", help="Recover one saved serial dump into a new stereo WAV")
    saved.add_argument("uart", type=Path)
    saved.add_argument("wav", type=Path)
    for mode in (live, saved):
        mode.add_argument("--elf", help="Expected full descriptor ELF SHA256 from the local build JSON")
    args = parser.parse_args()
    if args.mode == "record":
        report = record(args)
    else:
        with args.uart.open("rb") as uart:
            pcm, report = decode(uart, args.elf)
        write_wav(args.wav, pcm)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
