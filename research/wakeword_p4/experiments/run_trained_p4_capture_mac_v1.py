"""Run the authenticated trained probe in one verified development app slot.

No partition/bootloader/NVS writes. Always restore the fresh full-slot backup
after a probe write attempt. Raw captures, backups and device records are private.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import reduce_fp32_trained_p4_capture_mac_v1 as reducer


@dataclass
class Settings:
    port: str
    mac: str
    status_url: str
    app_sha256: str
    restore_app: Path
    restore_sha256: str
    app_offset: int
    slot: str
    slot_bytes: int
    partition_offset: int
    output: Path
    restore_readiness: str = "online"
    capture_timeout: int = 600
    baud: int = 1500000


def usb_serial_port(port, ports=None):
    """Validate the explicit callout port by USB metadata, not driver naming."""
    assert port.startswith("/dev/cu."), "An explicit macOS callout port is required"
    if ports is None:
        from serial.tools import list_ports
        ports = list_ports.comports()
    matches = [entry for entry in ports if entry.device == port]
    assert len(matches) == 1, "The explicit serial port must be currently enumerated"
    entry, = matches
    assert entry.vid is not None and entry.pid is not None, "The explicit port must be a USB device"
    return dict(device=entry.device, vid=entry.vid, pid=entry.pid,
                serial_number=entry.serial_number, location=entry.location)


def app_elf_sha256(image):
    # ESP image header24 + first segment header8 + esp_app_desc_t.
    assert len(image) >= 288 and image[0] == 0xe9
    assert struct.unpack_from("<I", image, 32)[0] == 0xabcd5432
    return image[32 + 144:32 + 176].hex()


def partition_entries(data):
    assert len(data) == 4096
    entries, hashed, checksum_seen, ended = [], bytearray(), False, False
    for offset in range(0, len(data), 32):
        record = data[offset:offset + 32]
        if record == b"\xff" * 32:
            ended = True
            assert all(b == 255 for b in data[offset:])
            break
        if record[:2] == b"\xeb\xeb":
            assert not checksum_seen and record[:16] == b"\xeb\xeb" + b"\xff" * 14
            assert record[16:] == hashlib.md5(hashed).digest()
            checksum_seen = True
            continue
        assert not checksum_seen
        magic, kind, subtype, start, size, label, flags = struct.unpack("<2sBBII16sI", record)
        assert magic == b"\xaa\x50" and start % 4096 == 0 and size > 0 and size % 4096 == 0
        assert flags == 0, "Encrypted/read-only partitions require a different experiment"
        name = label.split(b"\0", 1)[0].decode("ascii")
        assert name and all(e["label"] != name for e in entries)
        assert all(start + size <= e["offset"] or e["offset"] + e["size"] <= start for e in entries)
        entries.append(dict(type=kind, subtype=subtype, offset=start, size=size, label=name))
        hashed.extend(record)
    assert ended and checksum_seen and entries
    return entries


def ready(value, cfg, service=None):
    voice, ota = value["voice"], value["ota"]
    return (value["app_sha256"] == cfg.app_sha256 and ota["running"] == cfg.slot
            and ota["busy"] is False and ota.get("state") in ("valid", "initial")
            and voice["active"] is False and voice.get("busy", False) is False
            and voice["wake"]["listening"] is True
            and (service is None or voice["service_id"] == service)
            and (cfg.restore_readiness == "app-only" or value["nyx"]["state"] == "up"))


def snapshot(value):
    # Retain the fields needed for restoration without copying Wi-Fi names,
    # credential-bearing configuration or unrelated conversation state.
    return dict(app_sha256=value["app_sha256"], ota=value["ota"],
                voice={k: value["voice"][k] for k in ("active", "service_id", "wake")},
                nyx_state=value["nyx"]["state"])


@contextmanager
def finish_restoration():
    previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_transaction(cfg, backend, image, references, binding, *, capture_reducer=None, scope=None):
    report = dict(schema="mac-trained-p4-app-only-execution-v1", binding=binding,
                  runner_sha256=reducer.sha(Path(__file__).read_bytes()),
                  reducer_sha256=reducer.sha(Path(reducer.__file__).read_bytes()),
                  execution_parameters=dict(port=cfg.port, expected_mac=cfg.mac, status_url=cfg.status_url,
                      expected_app_sha256=cfg.app_sha256, app_offset=cfg.app_offset, slot=cfg.slot,
                      slot_bytes=cfg.slot_bytes, partition_table_offset=cfg.partition_offset,
                      restore_readiness=cfg.restore_readiness, baud=cfg.baud, capture_timeout=cfg.capture_timeout),
                  backend=backend.kind, capture_origin_bound=False, candidate_qualified=False,
                  write_attempted=False, restored=False, phases=[],
                  limits="Standalone original trained numerical subset only. Not full540, panel headroom, recognition/FAR, audio, first-utterance or long-call acceptance.")
    if scope is not None:
        report["limits"] = scope
    if capture_reducer is not None:
        report["capture_reducer_sha256"] = reducer.sha(Path(inspect.getsourcefile(capture_reducer)).read_bytes())
        report["capture_reducer_name"] = capture_reducer.__module__ + "." + capture_reducer.__name__
    service = None

    def phase(name):
        report["phases"].append(name)
        (cfg.output / "report-private.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(dict(phase=name)), flush=True)

    try:
        phase("preflight")
        assert cfg.restore_readiness in ("online", "app-only")
        assert re.fullmatch(r"[0-9a-f]{64}", cfg.app_sha256)
        assert re.fullmatch(r"[0-9a-f]{64}", cfg.restore_sha256)
        assert re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", cfg.mac)
        assert cfg.partition_offset >= 4096 and cfg.partition_offset % 4096 == 0
        assert cfg.app_offset >= cfg.partition_offset + 4096 and cfg.app_offset % 65536 == 0
        assert cfg.slot_bytes > 0 and cfg.slot_bytes % 4096 == 0
        assert re.fullmatch(r"ota_(?:[0-9]|1[0-5])", cfg.slot)
        assert 120 <= cfg.capture_timeout <= 900
        restore = cfg.restore_app.read_bytes()
        assert reducer.sha(restore) == cfg.restore_sha256
        assert app_elf_sha256(restore) == cfg.app_sha256
        assert 0 < len(image) <= cfg.slot_bytes and 0 < len(restore) <= cfg.slot_bytes
        assert reducer.sha(image) == binding["image_sha256"]
        backend.ensure_unowned()
        before = backend.status()
        assert ready(before, cfg)
        service = before["voice"]["service_id"]
        report["before"] = snapshot(before)
        assert backend.chip_mac("identity-before-private.log") == cfg.mac
        table = backend.read_flash(cfg.partition_offset, 4096, "partition-before-private.bin")
        entries = partition_entries(table)
        active, = [e for e in entries if e["label"] == cfg.slot]
        assert active == dict(type=0, subtype=0x10 + int(cfg.slot[4:]),
                              offset=cfg.app_offset, size=cfg.slot_bytes, label=cfg.slot)
        ota_entry, = [e for e in entries if e["type"] == 1 and e["subtype"] == 0]
        ota_data = backend.read_flash(ota_entry["offset"], ota_entry["size"], "otadata-before-private.bin")
        assert len(ota_data) == ota_entry["size"]
        backup = backend.read_flash(cfg.app_offset, cfg.slot_bytes, "original-slot-private.bin")
        assert len(backup) == cfg.slot_bytes and backup[:len(restore)] == restore
        assert app_elf_sha256(backup) == cfg.app_sha256
        report["original_slot_sha256"] = reducer.sha(backup)
        report["partition_sha256"] = reducer.sha(table)
        report["otadata_sha256"] = reducer.sha(ota_data)
        report["original_restore_app_sha256"] = reducer.sha(restore)
        report["slot"] = active
        current = backend.wait_status(cfg, service)
        assert ready(current, cfg, service)
        identity = backend.serial_status()
        assert identity["ip"] == backend.status_host_ip()
        assert identity["slug"] == current["nyx"]["slug"]
        assert identity["fw"] == current["fw"]
        report["serial_endpoint_identity"] = identity
        phase("backup_verified")
        # Set this before invoking the writer: a failed writer may have erased
        # or partially written the target slot and still requires restoration.
        report["write_attempted"] = True
        phase("probe_write_attempt")
        backend.write_app(image, "probe", reset=False)
        readback = backend.read_flash(cfg.app_offset, len(image), "probe-readback-private.bin", reset=False)
        assert readback == image
        report["probe_readback_sha256"] = reducer.sha(readback)
        phase("probe_readback_verified")
        raw = backend.capture(cfg.capture_timeout)
        report["capture_sha256"] = reducer.sha(raw)
        phase("capture_complete")
        report["reduction"] = (capture_reducer or reducer.reduce_capture)(raw, references)
        report["capture_origin_bound"] = backend.kind == "hardware"
        phase("protocol_validated")
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, detail=str(error))
        raise
    finally:
        try:
            if report["write_attempted"]:
                with finish_restoration():
                    phase("restoring_original_slot")
                    # The backend rechecks USB identity before every write.
                    backend.write_app(backup, "restore", reset=True)
                    actual = backend.read_flash(cfg.app_offset, cfg.slot_bytes, "restored-slot-private.bin")
                    assert actual == backup
                    actual_table = backend.read_flash(cfg.partition_offset, 4096, "partition-after-private.bin")
                    assert actual_table == table
                    actual_ota = backend.read_flash(ota_entry["offset"], ota_entry["size"], "otadata-after-private.bin")
                    assert actual_ota == ota_data
                    after = backend.wait_status(cfg, service)
                    assert ready(after, cfg, service)
                    report["after"] = snapshot(after)
                    report["restored"] = True
                    report["restored_slot_sha256"] = reducer.sha(actual)
                    report["restored_partition_sha256"] = reducer.sha(actual_table)
                    report["restored_otadata_sha256"] = reducer.sha(actual_ota)
                    phase("restoration_verified")
        except BaseException as cleanup_error:
            report["restoration_error"] = dict(type=type(cleanup_error).__name__, detail=str(cleanup_error))
            raise RuntimeError("Original development app restoration was not verified; retain private logs") from cleanup_error
        finally:
            (cfg.output / "report-private.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["restored"] and report["capture_origin_bound"] == (backend.kind == "hardware")
    return report


class HardwareBackend:
    kind = "hardware"

    def __init__(self, cfg):
        self.cfg = cfg

    def ensure_unowned(self):
        result = subprocess.run(["lsof", "-t", self.cfg.port], capture_output=True, text=True)
        assert result.returncode in (0, 1) and not result.stdout.strip(), "Serial port already has an owner"

    def status(self):
        with urllib.request.urlopen(self.cfg.status_url, timeout=5) as response:
            return json.load(response)

    def status_host_ip(self):
        return socket.gethostbyname(urllib.parse.urlsplit(self.cfg.status_url).hostname)

    def serial_status(self):
        import serial
        path = self.cfg.output / "original-console-identity-private.log"
        with serial.Serial(self.cfg.port, 115200, timeout=.2) as uart, path.open("wb") as log:
            uart.dtr = False
            uart.rts = False
            deadline, next_query, pending = time.monotonic() + 45, 0., bytearray()
            while time.monotonic() < deadline:
                if time.monotonic() >= next_query:
                    uart.write(b"\nnyx-status\n")
                    next_query = time.monotonic() + 3
                data = uart.read(4096)
                log.write(data)
                log.flush()
                pending.extend(data)
                assert len(pending) <= 65536
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    if b"api_key=" in line and b"nyx_link=" in line:
                        fields = dict(re.findall(r"(?:^|\s)(ip|slug|fw)=([^\s]+)", line.decode("utf-8", errors="replace")))
                        if set(fields) == {"ip", "slug", "fw"} and fields["ip"] != "none":
                            return fields
            raise TimeoutError("Original USB console endpoint identity")

    def wait_status(self, cfg, service):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                value = self.status()
                if ready(value, cfg, service):
                    return value
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1)
        raise TimeoutError("Original app/service/idle wake readiness")

    def esptool(self, arguments, log, reset=True):
        command = [sys.executable, "-m", "esptool", "--chip", "esp32p4", "--port", self.cfg.port,
                   "--baud", str(self.cfg.baud), "--before", "default_reset", "--after",
                   "hard_reset" if reset else "no_reset", *arguments]
        try:
            result = subprocess.run(command, capture_output=True, timeout=240)
        except subprocess.TimeoutExpired as error:
            (self.cfg.output / log).write_bytes((error.stdout or b"") + (error.stderr or b""))
            raise
        (self.cfg.output / log).write_bytes(result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout.decode("utf-8", errors="replace")

    def chip_mac(self, log):
        text = self.esptool(["chip_id"], log)
        values = {s.lower() for s in re.findall(r"(?i)\bMAC:\s*((?:[0-9a-f]{2}:){5}[0-9a-f]{2})", text)}
        assert len(values) == 1
        return values.pop()

    def read_flash(self, offset, size, name, reset=True):
        path = self.cfg.output / name
        self.esptool(["read_flash", hex(offset), hex(size), str(path)], name + ".log", reset)
        data = path.read_bytes()
        assert len(data) == size
        return data

    def write_app(self, data, label, reset):
        assert self.chip_mac(label + "-identity-private.log") == self.cfg.mac
        assert 0 < len(data) <= self.cfg.slot_bytes
        path = self.cfg.output / (label + "-app-private.bin")
        path.write_bytes(data)
        self.esptool(["write_flash", hex(self.cfg.app_offset), str(path)], label + "-flash-private.log", reset)

    def capture(self, timeout):
        import serial
        path = self.cfg.output / "uart-private.log"
        # Open before resetting; retain bytes across short reads. read_until
        # may return an incomplete line on timeout, so frame in our own FIFO.
        with serial.Serial(self.cfg.port, 115200, timeout=.2) as uart, path.open("wb") as log:
            uart.dtr = False
            uart.rts = True
            time.sleep(.1)
            uart.reset_input_buffer()
            uart.rts = False
            pending = bytearray()
            deadline, last_progress = time.monotonic() + timeout, time.monotonic()
            while time.monotonic() < deadline:
                data = uart.read(4096)
                if data:
                    log.write(data)
                    log.flush()
                    pending.extend(data)
                    assert len(pending) <= 65536, "UART line exceeded protocol limit"
                    while b"\n" in pending:
                        line, _, remainder = pending.partition(b"\n")
                        pending = bytearray(remainder)
                        if line.startswith(b"P4_TRAINED_DONE "):
                            return path.read_bytes()
                        if any(marker in line for marker in (b"P4_TRAINED_ERROR", b"Guru Meditation", b"abort()")):
                            raise RuntimeError("Trained probe firmware failure; see private capture")
                if time.monotonic() - last_progress >= 15:
                    print(json.dumps(dict(phase="uart_capture", bytes=log.tell())), flush=True)
                    last_progress = time.monotonic()
            raise TimeoutError("Trained P4 capture deadline")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "run-root", "restore-app", "output"):
        ap.add_argument("--" + name, type=Path, required=True)
    for name in ("port", "expected-mac", "status-url", "expected-app-sha256", "expected-restore-sha256", "slot"):
        ap.add_argument("--" + name, required=True)
    for name in ("app-offset", "slot-bytes", "partition-table-offset"):
        ap.add_argument("--" + name, type=lambda s: int(s, 0), required=True)
    ap.add_argument("--restore-readiness", choices=("online", "app-only"), default="online")
    ap.add_argument("--capture-timeout", type=int, default=600)
    ap.add_argument("--baud", type=int, choices=(115200, 921600, 1500000), default=1500000)
    a = ap.parse_args()
    url = urllib.parse.urlsplit(a.status_url)
    assert url.scheme == "http" and url.hostname and not url.username and not url.password
    usb_serial_port(a.port)
    references, binding = reducer.load_fixtures(a.repository_root, a.run_root)
    image = (a.run_root / "build/nyx_trained_fp32_probe.bin").read_bytes()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = Settings(a.port, a.expected_mac.lower(), a.status_url, a.expected_app_sha256,
                   a.restore_app, a.expected_restore_sha256, a.app_offset, a.slot, a.slot_bytes,
                   a.partition_table_offset, a.output, a.restore_readiness, a.capture_timeout, a.baud)

    def interrupted(_sig, _frame):
        raise KeyboardInterrupt("Interrupted; restoring original development app")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        run_transaction(cfg, HardwareBackend(cfg), image, references, binding)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
