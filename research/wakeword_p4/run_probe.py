#!/usr/bin/env python3
"""Run a numeric-only P4 probe in a verified active app slot, then restore it.

Use a development panel with no other USB owner. Outputs belong outside git.
This does not measure acoustic recognition or full-application resource use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=1500000,
                        choices=(115200, 230400, 460800, 921600, 1500000))
    parser.add_argument("--frontend-witness", action="store_true",
                        help="Collect numeric frontend table/stream hashes instead of model probe output")
    parser.add_argument("--expected-mac", required=True)
    parser.add_argument("--status-url", required=True)
    parser.add_argument("--restore-app", type=Path, required=True)
    parser.add_argument("--expected-app-sha256", required=True)
    parser.add_argument("--app-offset", type=lambda s: int(s, 0), required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--slot-bytes", type=lambda s: int(s, 0), required=True)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--vectors-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.status_url.startswith("http://"):
        parser.error("Expected the development panel's explicit LAN HTTP status URL")
    probe_hash = hashlib.sha256(args.probe_bin.read_bytes()).hexdigest()
    if probe_hash != args.expected_probe_sha256:
        raise ValueError("Probe binary identity mismatch")
    manifest = json.loads(args.vectors_manifest.read_text())
    if bytes.fromhex(args.expected_app_sha256) not in args.restore_app.read_bytes():
        raise ValueError("Restore app does not contain the expected ELF identity")
    if not all(0 < p.stat().st_size <= args.slot_bytes for p in (args.restore_app, args.probe_bin)):
        raise ValueError("An app exceeds the explicitly verified slot size")

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def esptool(arguments, log, timeout=120):
        value = subprocess.run([sys.executable, "-m", "esptool", "--chip", "esp32p4", "--port", args.port,
                                *arguments], capture_output=True, text=True, timeout=timeout)
        (args.output/log).write_text(value.stdout+value.stderr)
        value.check_returncode()
        return value.stdout

    def flash(path, log):
        esptool(["--baud", str(args.baud), "--before", "default_reset", "--after", "hard_reset",
                 "write_flash", hex(args.app_offset), str(path)], log)

    before = status()
    if before["app_sha256"] != args.expected_app_sha256 or before["ota"]["running"] != args.slot:
        raise ValueError("Development app identity or active slot mismatch")
    if before["voice"]["active"] or before["ota"]["busy"] or before["nyx"]["state"] != "up":
        raise ValueError("Development app is busy or its service is not connected")
    chip = esptool(["chip_id"], "chip-private.log", 30)
    if args.expected_mac.lower() not in chip.lower():
        raise ValueError("USB chip does not match the development panel")
    result = {"probe_bin_sha256": probe_hash, "probe_bin_bytes": args.probe_bin.stat().st_size,
              "vectors_manifest_sha256": hashlib.sha256(args.vectors_manifest.read_bytes()).hexdigest(),
              "vectors": manifest, "models": [], "frontend_witnesses": [], "frontend_deltas": [],
              "scope": "Standalone numeric compatibility/resources only; no recognition acceptance"}

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Probe interrupted; restoring original app")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        flash(args.probe_bin, "probe-flash-private.log")
        with serial.Serial(args.port, 115200, timeout=.2) as uart, (args.output/"uart-private.log").open("w") as log:
            uart.dtr = False
            uart.rts = True
            time.sleep(.1)
            uart.rts = False
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                line = uart.read_until(b"\n", 8192).decode("utf-8", errors="replace").strip()
                if line:
                    log.write(line+"\n")
                    log.flush()
                markers = ("P4_FRONTEND_TABLE|P4_FRONTEND_WITNESS|P4_FRONTEND_DELTA|P4_FRONTEND_DONE"
                           if args.frontend_witness else "PROBE_FRONTEND|PROBE_MODEL|P4_WAKE_PROBE_DONE")
                match = re.search(r"("+markers+r") (\{.*\})", line)
                if match:
                    marker, data = match.groups()
                    value = json.loads(data)
                    if marker == "PROBE_MODEL":
                        result["models"].append(value)
                    elif marker == "P4_FRONTEND_WITNESS":
                        result["frontend_witnesses"].append(value)
                    elif marker == "P4_FRONTEND_DELTA":
                        result["frontend_deltas"].append(value)
                    else:
                        result[marker] = value
                    print(match.group(0), flush=True)
                    if marker in ("P4_WAKE_PROBE_DONE", "P4_FRONTEND_DONE"):
                        break
                if "abort()" in line or "Guru Meditation" in line:
                    raise RuntimeError("Probe firmware failed; see private UART log")
            if ("P4_FRONTEND_DONE" if args.frontend_witness else "P4_WAKE_PROBE_DONE") not in result:
                raise TimeoutError("Numeric probe did not complete")
    finally:
        # A normal cancellation still restores. Do not SIGKILL or remove power.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("Restoring original development app", flush=True)
        flash(args.restore_app, "restore-private.log")
        deadline = time.monotonic()+60
        after = None
        while time.monotonic() < deadline:
            try:
                after = status()
                if (after["app_sha256"] == args.expected_app_sha256 and after["nyx"]["state"] == "up"
                        and after["voice"]["wake"]["listening"]):
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        if not after or after["app_sha256"] != args.expected_app_sha256 or after["ota"]["running"] != args.slot:
            raise RuntimeError("RESTORE VERIFICATION FAILED")
        if (after["nyx"]["state"] != "up" or not after["voice"]["wake"]["listening"]
                or after["voice"]["service_id"] != before["voice"]["service_id"]):
            raise RuntimeError("App restored, but original service/wake state was not recovered")
        result["restored"] = {"app_sha256": after["app_sha256"], "nyx_state": "up", "wake_listening": True}
        (args.output/"report.json").write_text(json.dumps(result, indent=2)+"\n")
        print("Original app, service selection and Hi ESP listening verified", flush=True)

    if args.frontend_witness:
        expected = manifest['witnesses']
        if (len(result['frontend_witnesses']) != len(expected)
                or result['P4_FRONTEND_DONE'] != {'streams': len(expected), 'all_inputs_consumed': True}
                or 'P4_FRONTEND_TABLE' not in result):
            raise RuntimeError('Incomplete frontend witness capture')
        for actual, row in zip(result['frontend_witnesses'], expected, strict=True):
            if (actual['pack_index'], actual['frames']) != (row['pack_index'], row['frames']):
                raise RuntimeError('Frontend witness identity or shape mismatch')
            if not re.fullmatch('[0-9a-f]{64}', actual['raw_sha256']):
                raise RuntimeError('Malformed frontend output hash')
        return  # Numerical differences are observations, not an acceptance gate.

    flags = ("frontend", "pretrained_model", "mac_exported_model", "frontend_to_pretrained", "frontend_to_mac_exported")
    if not all(result["P4_WAKE_PROBE_DONE"][k] for k in flags) or len(result["models"]) != 4:
        raise RuntimeError("Numeric probe reported a failed check")
    for measured in result["models"]:
        if measured["calls"] != manifest["frames"]:
            raise RuntimeError("Numeric vector count mismatch")
        if not measured["board_frontend"] and measured["mismatches"]:
            raise RuntimeError("P4 model outputs differ from host BUILTIN_REF goldens")


if __name__ == "__main__":
    main()
