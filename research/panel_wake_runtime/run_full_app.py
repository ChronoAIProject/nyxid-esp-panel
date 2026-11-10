"""Temporarily test a full experimental app on one verified development panel.

Only the explicitly identified active app slot is written. A normal failure or
SIGTERM restores the original app and verifies its service/wake state. Never
SIGKILL this process or remove power during restoration. Logs stay private.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def run_acceptance(command, log, timeout):
    # Composite checks can have an acoustic child with its own call cleanup.
    # Terminate the isolated group gracefully before restoring/flashing audio.
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    try:
        returncode = process.wait(timeout=timeout)
        result = subprocess.CompletedProcess(command, returncode)
        result.check_returncode()
        return result
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                process.poll()  # Reap the leader while its children clean up.
                try:
                    os.killpg(process.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(.1)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            process.wait(timeout=5)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True)
    p.add_argument("--expected-mac", required=True)
    p.add_argument("--status-url", required=True)
    p.add_argument("--restore-app", type=Path, required=True)
    p.add_argument("--expected-restore-elf-sha256", required=True)
    p.add_argument("--experimental-app", type=Path, required=True)
    p.add_argument("--expected-experimental-elf-sha256", required=True)
    p.add_argument("--app-offset", type=lambda s: int(s, 0), required=True)
    p.add_argument("--slot", required=True)
    p.add_argument("--slot-bytes", type=lambda s: int(s, 0), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--command-timeout", type=int, default=300)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not args.status_url.startswith("http://"):
        p.error("Supply an explicit development LAN status URL and acceptance command after --")
    for app, digest in ((args.restore_app, args.expected_restore_elf_sha256),
                        (args.experimental_app, args.expected_experimental_elf_sha256)):
        data = app.read_bytes()
        if not 0 < len(data) <= args.slot_bytes or bytes.fromhex(digest) not in data:
            raise ValueError("App slot size or embedded ELF identity mismatch")
    args.output.mkdir(parents=True, exist_ok=False)

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def esptool(arguments, filename):
        result = subprocess.run([sys.executable, "-m", "esptool", "--chip", "esp32p4", "--port", args.port,
            *arguments], capture_output=True, text=True, timeout=150)
        (args.output / filename).write_text(result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    def flash(path, filename):
        esptool(["--baud", "1500000", "--before", "default_reset", "--after", "hard_reset",
                 "write_flash", hex(args.app_offset), str(path)], filename)

    before = status()
    if (before["app_sha256"] != args.expected_restore_elf_sha256
            or before["ota"]["running"] != args.slot or before["ota"]["busy"]
            or before["voice"]["busy"] or before["nyx"]["state"] != "up"
            or not before["voice"]["wake"]["listening"]):
        raise ValueError("Original development app, slot, connection or idle wake state differs")
    chip = esptool(["chip_id"], "chip-private.log")
    if args.expected_mac.lower() not in chip.lower():
        raise ValueError("USB device does not match the expected development panel")
    (args.output / "before-private.json").write_text(json.dumps(before, indent=2) + "\n")
    report = {"scope": "Temporary full-app development experiment; not a production release",
              "experimental_bin_sha256": hashlib.sha256(args.experimental_app.read_bytes()).hexdigest(),
              "experimental_elf_sha256": args.expected_experimental_elf_sha256,
              "restored": False}

    def wait_ready(digest, word):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                value = status()
                if (value["app_sha256"] == digest and value["voice"]["wake"]["word"] == word
                        and value["voice"]["wake"]["state"] in ("unavailable", "error")):
                    raise RuntimeError("Requested app booted but its wake detector failed")
                if (value["app_sha256"] == digest and value["ota"]["running"] == args.slot
                        and value["nyx"]["state"] == "up" and value["voice"]["wake"]["listening"]
                        and value["voice"]["wake"]["word"] == word
                        and value["voice"]["service_id"] == before["voice"]["service_id"]):
                    return value
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise TimeoutError("App identity, service selection or wake listener did not recover")

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("Interrupted; restoring the original development app")

    signal.signal(signal.SIGTERM, interrupted)
    monitor = None
    monitor_log = None
    try:
        flash(args.experimental_app, "experimental-flash-private.log")
        # The flash subprocess has released USB. Keep a single UART reader
        # during acceptance and close it before the restoration flash.
        monitor_log = (args.output / "monitor-private.log").open("w")
        monitor = subprocess.Popen([sys.executable,
            str(Path(__file__).resolve().parents[2] / "tools" / "monitor_uart.py"),
            "--port", args.port, "--output", str(args.output / "uart-private.log")],
            stdout=monitor_log, stderr=subprocess.STDOUT)
        experimental = wait_ready(args.expected_experimental_elf_sha256, "Hey Nyx")
        (args.output / "experimental-private.json").write_text(json.dumps(experimental, indent=2) + "\n")
        print("Experimental full app, original service selection and Hey Nyx listening verified", flush=True)
        with (args.output / "acceptance-private.log").open("w") as log:
            result = run_acceptance(command, log, args.command_timeout)
        report["acceptance_exit_code"] = result.returncode
        result.check_returncode()
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if monitor:
            if monitor.poll() is None:
                monitor.send_signal(signal.SIGINT)
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
        if monitor_log:
            monitor_log.close()
        print("Restoring original development app", flush=True)
        try:
            flash(args.restore_app, "restore-flash-private.log")
            after = wait_ready(args.expected_restore_elf_sha256, before["voice"]["wake"]["word"])
            (args.output / "restored-private.json").write_text(json.dumps(after, indent=2) + "\n")
            report["restored"] = True
            print("Original app, original service and Hi ESP listener restored and verified", flush=True)
        finally:
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
