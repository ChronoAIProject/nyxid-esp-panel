#!/usr/bin/env python3
"""Provision a panel from an encrypted Nyx node-agent config, without printing secrets.

Requires pyserial and cryptography. Close any serial monitor first.
"""
import argparse
import base64
from pathlib import Path
import time
import tomllib

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import serial


def credentials(root):
    cfg = tomllib.loads((root / "config.toml").read_text())
    aes = AESGCM((root / ".keyfile").read_bytes())

    def decrypt(value):
        raw = base64.b64decode(value)
        return aes.decrypt(raw[:12], raw[12:], None).decode()

    return cfg["node"]["id"], decrypt(cfg["node"]["auth_token_encrypted"]), decrypt(
        cfg["signing"]["shared_secret_encrypted"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--port", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--monitor-seconds", type=int, default=0)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    node, token, secret = credentials(args.config_dir)
    commands = [("nyx-key", token), ("nyx-node", node),
                ("nyx-secret", secret), ("nyx-slug", args.slug)]
    for _, value in commands:
        if not value or any(c.isspace() for c in value):
            raise SystemExit("Invalid provisioning value")
    port = serial.Serial(port=None, baudrate=115200, timeout=0.2)
    port.dtr = port.rts = False
    port.port = args.port
    port.open()
    try:
        # Some USB-UART bridges pulse reset on open even with DTR/RTS disabled.
        # Allow BSP, display and the console to finish booting before typing.
        deadline = time.monotonic() + 25
        boot = b""
        while time.monotonic() < deadline:
            chunk = port.read(4096)
            boot += chunk
            if b"\x1b[6n" in chunk:
                port.write(b"\x1b[1;80R")
            if b"panel>" in boot:
                break
        port.reset_input_buffer()
        for command, value in commands:
            port.write(f"{command} {value}\r".encode())
            deadline = time.monotonic() + 3
            reply = b""
            while time.monotonic() < deadline:
                reply += port.read(4096)
                if b"\x1b[6n" in reply[-64:]:
                    port.write(b"\x1b[1;80R")
                # ESP console redraws/echoes its prompt; never forward this buffer.
                if b"Set; reboot to apply" in reply:
                    break
            if b"Failed" in reply or b"Invalid" in reply or b"Usage:" in reply:
                raise SystemExit(f"Provisioning command failed: {command}")
            if b"Set; reboot to apply" not in reply and command != "nyx-secret":
                raise SystemExit(f"No acknowledgement: {command}")
            print(f"Saved {command}", flush=True)
        port.write(b"reboot\r\n")
        port.flush()
        print(f"Provisioned node {node}; reboot requested")
        if args.monitor_seconds:
            if not args.log_file:
                raise SystemExit("--monitor-seconds requires --log-file")
            args.log_file.touch(mode=0o600, exist_ok=True)
            args.log_file.chmod(0o600)
            with args.log_file.open("wb") as log:
                deadline = time.monotonic() + args.monitor_seconds
                while time.monotonic() < deadline:
                    log.write(port.read(4096))
                    log.flush()
            print("Boot/runtime log captured privately")
    finally:
        port.close()


if __name__ == "__main__":
    main()
