"""Exercise app-only transaction failures with constructed flash/serial backends."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import check_trained_p4_capture_reducer_mac_v1 as capture_checks
import reduce_fp32_trained_p4_capture_mac_v1 as reducer
import run_trained_p4_capture_mac_v1 as runner


def make_table(offset, size):
    row = struct.pack("<2sBBII16sI", b"\xaa\x50", 1, 0, 0x9000, 8192, b"otadata", 0)
    row += struct.pack("<2sBBII16sI", b"\xaa\x50", 0, 0x10, offset, size, b"ota_0", 0)
    table = row + b"\xeb\xeb" + b"\xff" * 14 + hashlib.md5(row).digest()
    return table + b"\xff" * (4096 - len(table))


class FakeBackend:
    kind = "constructed"

    def __init__(self, cfg, image, raw, fail=None):
        self.cfg, self.image, self.raw, self.fail = cfg, image, raw, fail
        self.table = make_table(cfg.app_offset, cfg.slot_bytes)
        if fail == "wrong_partition_offset":
            self.table = make_table(cfg.app_offset + 65536, cfg.slot_bytes)
        if fail == "wrong_partition_size":
            self.table = make_table(cfg.app_offset, cfg.slot_bytes - 4096)
        self.original = cfg.restore_app.read_bytes() + b"\xff" * (cfg.slot_bytes - cfg.restore_app.stat().st_size)
        self.flash = self.original
        self.calls, self.probe_attempts, self.restore_attempts = [], 0, 0
        self.value = dict(fw="constructed", app_sha256=cfg.app_sha256,
                          ota=dict(running=cfg.slot, busy=False, state="valid"),
                          voice=dict(active=False, busy=False, service_id="constructed-service", wake=dict(listening=True)),
                          nyx=dict(state="up", slug="constructed-slug"))

    def ensure_unowned(self):
        self.calls.append("unowned")
        if self.fail == "serial_owner":
            raise AssertionError("constructed owner")

    def status(self):
        value = copy.deepcopy(self.value)
        if self.fail == "busy_voice":
            value["voice"]["active"] = True
        if self.fail == "pending_ota":
            value["ota"]["state"] = "pending_verify"
        if self.fail == "wrong_http_slot":
            value["ota"]["running"] = "ota_1"
        return value

    def wait_status(self, cfg, service):
        self.calls.append("wait_status")
        if self.restore_attempts and self.fail == "restore_readiness":
            raise TimeoutError("constructed restore status timeout")
        return self.status()

    def chip_mac(self, log):
        self.calls.append("chip_mac")
        return "00:00:00:00:00:02" if self.fail == "wrong_usb_mac" else self.cfg.mac

    def status_host_ip(self):
        return "192.0.2.1"

    def serial_status(self):
        self.calls.append("serial_status")
        value = dict(ip="192.0.2.1", slug="constructed-slug", fw="constructed")
        if self.fail == "wrong_serial_ip":
            value["ip"] = "192.0.2.2"
        if self.fail == "wrong_serial_slug":
            value["slug"] = "another-board"
        if self.fail == "wrong_serial_firmware":
            value["fw"] = "another-app"
        return value

    def read_flash(self, offset, size, name, reset=True):
        self.calls.append("read:" + name)
        if offset == 0x9000:
            data = b"\x55" * size
            if self.fail == "changed_otadata_after" and self.restore_attempts:
                data = b"\x56" + data[1:]
            return data
        if offset == self.cfg.partition_offset:
            data = self.table
            if self.fail == "bad_table_md5" and not self.restore_attempts:
                data = data[:4] + bytes([data[4] ^ 1]) + data[5:]
            if self.fail == "changed_partition_after" and self.restore_attempts:
                data = make_table(self.cfg.app_offset + 65536, self.cfg.slot_bytes)
            return data
        assert offset == self.cfg.app_offset
        data = self.flash[:size]
        if ((self.fail == "bad_original_backup" and not self.probe_attempts)
                or (self.fail == "bad_probe_readback" and self.probe_attempts and not self.restore_attempts)
                or (self.fail == "bad_restore_readback" and self.restore_attempts)):
            data = bytes([data[0] ^ 1]) + data[1:]
        return data

    def write_app(self, data, label, reset):
        self.calls.append("write:" + label)
        if label == "probe":
            self.probe_attempts += 1
            self.flash = data + self.flash[len(data):]
            if self.fail == "partial_probe_write":
                raise OSError("constructed partial write")
        else:
            assert label == "restore"
            self.restore_attempts += 1
            if self.fail == "restore_write":
                raise OSError("constructed restore write failure")
            self.flash = data

    def capture(self, timeout):
        self.calls.append("capture")
        if self.fail == "capture_timeout":
            raise TimeoutError("constructed capture timeout")
        if self.fail == "capture_interrupt":
            raise KeyboardInterrupt("constructed cancellation")
        if self.fail == "invalid_protocol":
            return self.raw.replace(b"P4_TRAINED_STEP", b"P4_TRAINED_ERROR", 1)
        return self.raw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("repository-root", "run-root", "output-dir"):
        ap.add_argument("--" + name, type=Path, required=True)
    a = ap.parse_args()
    a.output_dir.mkdir(exist_ok=False)
    # WCH, CP210x and CDC drivers use different device names. Only a freshly
    # enumerated USB callout port is eligible; the name alone proves nothing.
    usb_port_cases = []
    for name in ("wchusbserial-CONSTRUCTED", "SLAB_USBtoUART-CONSTRUCTED", "usbmodem-CONSTRUCTED"):
        port = "/dev/cu." + name
        entry = SimpleNamespace(device=port, vid=0x1a86, pid=0x55d3,
                                serial_number="CONSTRUCTED", location="0-1")
        assert runner.usb_serial_port(port, [entry])["device"] == port
        usb_port_cases.append(dict(case=name, accepted=True))
    for name, entries in (
            ("unlisted", []),
            ("usb-in-name-only", [SimpleNamespace(device="/dev/cu.usb-in-name-only", vid=None, pid=None)]),
            ("Bluetooth-Incoming-Port", [SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", vid=None, pid=None)])):
        try:
            runner.usb_serial_port("/dev/cu." + name, entries)
        except AssertionError:
            usb_port_cases.append(dict(case=name, accepted=False))
        else:
            raise AssertionError("Non-USB or absent port accepted")
    refs, binding = reducer.load_fixtures(a.repository_root, a.run_root)
    image = (a.run_root / "build/nyx_trained_fp32_probe.bin").read_bytes()
    assert runner.app_elf_sha256(image) == reducer.sha((a.run_root / "build/nyx_trained_fp32_probe.elf").read_bytes())
    raw = capture_checks.encode(capture_checks.construct(refs))
    restore = bytearray(4096)
    restore[0] = 0xe9
    struct.pack_into("<I", restore, 32, 0xabcd5432)
    restore[176:208] = bytes.fromhex("11" * 32)
    restore_path = a.output_dir / "constructed-restore.bin"
    restore_path.write_bytes(restore)
    cases = []
    preflight = {"serial_owner", "busy_voice", "pending_ota", "wrong_http_slot", "wrong_usb_mac",
                 "wrong_serial_ip", "wrong_serial_slug", "wrong_serial_firmware", "bad_table_md5", "bad_original_backup",
                 "wrong_partition_offset", "wrong_partition_size", "online_disconnected"}
    failed_cleanup = {"restore_write", "bad_restore_readback", "changed_partition_after", "changed_otadata_after", "restore_readiness"}
    during_probe = {"partial_probe_write", "bad_probe_readback", "capture_timeout", "capture_interrupt", "invalid_protocol"}
    for name in ["success", "app_only_connecting"] + sorted(preflight | failed_cleanup | during_probe):
        out = a.output_dir / name
        out.mkdir()
        cfg = runner.Settings("/dev/cu.usb-CONSTRUCTED", "00:00:00:00:00:01", "http://192.0.2.1/status",
                              "11" * 32, restore_path, reducer.sha(restore), 0x10000, "ota_0", 0x200000,
                              0x8000, out)
        fake = FakeBackend(cfg, image, raw, None if name == "success" else name)
        if name in ("app_only_connecting", "online_disconnected"):
            fake.value["nyx"]["state"] = "connecting"
        if name == "app_only_connecting":
            cfg.restore_readiness = "app-only"
        error_type = None
        try:
            report = runner.run_transaction(cfg, fake, image, refs, binding)
        except BaseException as error:
            error_type = type(error).__name__
            report = json.loads((out / "report-private.json").read_text())
        assert report["backend"] == "constructed" and not report["capture_origin_bound"]
        if name in ("success", "app_only_connecting"):
            assert error_type is None and report["restored"] and fake.flash == fake.original
            assert report["reduction"]["invocations"] == 8336
        elif name in preflight:
            assert error_type is not None and not fake.probe_attempts and not fake.restore_attempts
            assert not report["write_attempted"] and fake.flash == fake.original
        elif name in during_probe:
            assert error_type is not None and fake.probe_attempts == fake.restore_attempts == 1
            assert report["restored"] and fake.flash == fake.original
            assert fake.calls.index("write:restore") > fake.calls.index("write:probe")
        else:
            assert error_type == "RuntimeError" and fake.probe_attempts == fake.restore_attempts == 1
            assert not report["restored"] and "restoration_error" in report
        cases.append(dict(case=name, error_type=error_type, probe_attempts=fake.probe_attempts,
                          restore_attempts=fake.restore_attempts, restored_verified=report["restored"], passed=True))

    # Other standalone probes reuse this exact backup/restore transaction.
    # A reducer error must still restore, with its original failure retained.
    for reject in (False, True):
        out = a.output_dir / ("custom_reducer_error" if reject else "custom_reducer")
        out.mkdir()
        cfg.output = out
        fake = FakeBackend(cfg, image, b"OTHER_PROBE_DONE\n")
        calls = []
        def other_reducer(captured, references):
            calls.append((captured, references))
            if reject:
                raise ValueError("constructed alternate protocol rejection")
            return dict(alternate_protocol=True)
        try:
            report = runner.run_transaction(cfg, fake, image, refs, binding,
                capture_reducer=other_reducer, scope="Constructed alternate probe")
        except ValueError:
            assert reject
            report = json.loads((out / "report-private.json").read_text())
        else:
            assert not reject and report["reduction"] == dict(alternate_protocol=True)
        assert calls == [(b"OTHER_PROBE_DONE\n", refs)]
        assert report["restored"] and fake.flash == fake.original
        assert report["limits"] == "Constructed alternate probe"
        assert report["capture_reducer_sha256"] == reducer.sha(Path(__file__).read_bytes())
        cases.append(dict(case=out.name, passed=True, restored_verified=True))

    # Exercise actual hardware-writer dispatch without invoking esptool: one
    # app address only, and zero write commands when USB identity differs.
    hardware = runner.HardwareBackend(cfg)
    commands = []
    hardware.chip_mac = lambda log: cfg.mac
    hardware.esptool = lambda args, log, reset=True: commands.append(args)
    hardware.write_app(image, "constructed-dispatch", reset=False)
    assert len(commands) == 1 and commands[0][:2] == ["write_flash", hex(cfg.app_offset)]
    hardware.chip_mac = lambda log: "00:00:00:00:00:02"
    try:
        hardware.write_app(image, "constructed-wrong-usb", reset=False)
    except AssertionError:
        pass
    else:
        raise AssertionError("Foreign USB identity reached writer")
    assert len(commands) == 1

    # Exercise the actual UART FIFO with arbitrary chunk boundaries and verify
    # the port closes on both a valid terminal marker and a firmware error.
    closed = []
    class Serial:
        def __init__(self, *args, **kwargs):
            self.parts = [raw[i:i + 511] for i in range(0, len(raw), 511)]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
        def reset_input_buffer(self):
            pass
        def read(self, size):
            return self.parts.pop(0) if self.parts else b""
    saved = sys.modules.get("serial")
    sys.modules["serial"] = SimpleNamespace(Serial=Serial)
    try:
        captured = runner.HardwareBackend(cfg).capture(3)
        assert captured == raw and closed == [True]
        class ErrorSerial(Serial):
            def __init__(self, *args, **kwargs):
                self.parts = [b"P4_TRAINED_ER", b'ROR {"case":0}\n']
        sys.modules["serial"] = SimpleNamespace(Serial=ErrorSerial)
        try:
            runner.HardwareBackend(cfg).capture(3)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Firmware error was not raised")
        assert closed == [True, True]
    finally:
        if saved is None:
            del sys.modules["serial"]
        else:
            sys.modules["serial"] = saved
    result = dict(schema="mac-trained-p4-app-only-runner-constructed-checks-v1",
                  source_sha256=reducer.sha(Path(__file__).read_bytes()),
                  runner_sha256=reducer.sha(Path(runner.__file__).read_bytes()),
                  reducer_sha256=reducer.sha(Path(reducer.__file__).read_bytes()),
                  binding=binding, cases=cases, all_passed=True,
                  usb_port_cases=usb_port_cases,
                  original_linked_app_descriptor_matches_actual_elf=True,
                  actual_writer_only_targets_verified_app_offset=True,
                  actual_writer_rejects_foreign_usb_before_command=True,
                  actual_capture_fifo_preserves_split_lines_and_closes_on_success_and_error=True,
                  hardware_accessed=False, actual_p4_invocations=0,
                  limits="Constructed flash/status/serial and invented timings only. Exercises transaction failure/restoration order and actual dispatch/FIFO helpers; no physical device, actual resource, acoustic recognition or candidate acceptance evidence.")
    (a.output_dir / "constructed-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(cases=len(cases), all_passed=True, hardware_accessed=False)))


if __name__ == "__main__":
    main()
