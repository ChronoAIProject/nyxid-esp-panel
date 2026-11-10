#!/usr/bin/env python3
"""Temporarily run a numeric head probe on an explicitly verified development P4.

Writes only the specified active app slot and always attempts original-app
restoration. Run without another UART owner; do not interrupt power or SIGKILL.
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


MODES = ('internal64k_model_flash', 'internal64k_model_psram', 'psram64k_model_psram')


def validate_result(result, manifest):
    rows = result.get('heads', [])
    if len(rows) != len(MODES) or {row.get('name') for row in rows} != set(MODES):
        raise ValueError('Missing, duplicate or unexpected head placement results')
    for row in rows:
        if (row.get('calls') != manifest['cases'] or row.get('mismatches') != 0
                or row.get('max_output_delta') != 0 or row.get('passed') is not True
                or not 0 < row.get('arena_used', 0) <= row.get('arena_allocated', 0)
                or not 0 < row.get('p50_us', 0) <= row.get('p95_us', 0) <= row.get('max_us', 0)):
            raise ValueError('A placement failed its exact-byte, call-count or resource gate')
    if result.get('done') != {'completed': True, 'numeric_passed': True}:
        raise ValueError('Firmware did not confirm completion and numerical fidelity')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', required=True)
    p.add_argument('--expected-mac', required=True)
    p.add_argument('--status-url', required=True)
    p.add_argument('--restore-app', type=Path, required=True)
    p.add_argument('--expected-restore-bin-sha256', required=True)
    p.add_argument('--expected-app-sha256', required=True)
    p.add_argument('--app-offset', type=lambda s: int(s, 0), required=True)
    p.add_argument('--slot', required=True)
    p.add_argument('--slot-bytes', type=lambda s: int(s, 0), required=True)
    p.add_argument('--probe-bin', type=Path, required=True)
    p.add_argument('--expected-probe-sha256', required=True)
    p.add_argument('--vectors-manifest', type=Path, required=True)
    p.add_argument('--expected-vectors-manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not args.status_url.startswith('http://') or not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', args.expected_mac):
        p.error('Specify the development LAN HTTP status URL and exact chip MAC')
    if args.app_offset <= 0 or args.app_offset % 0x10000 or args.slot_bytes <= 0 or args.slot not in ('ota_0', 'ota_1'):
        p.error('Specify the independently verified aligned active OTA app slot')
    for path, expected in ((args.restore_app, args.expected_restore_bin_sha256),
                           (args.probe_bin, args.expected_probe_sha256),
                           (args.vectors_manifest, args.expected_vectors_manifest_sha256)):
        if sha(path) != expected:
            raise ValueError('Artifact identity mismatch before any USB operation')
    if bytes.fromhex(args.expected_app_sha256) not in args.restore_app.read_bytes():
        raise ValueError('Restore image does not contain expected app ELF identity')
    if not all(0 < x.stat().st_size <= args.slot_bytes for x in (args.restore_app, args.probe_bin)):
        raise ValueError('Image exceeds verified active slot size')
    manifest = json.loads(args.vectors_manifest.read_text())
    if manifest.get('expected_modes') != list(MODES) or not isinstance(manifest.get('cases'), int) or not 0 < manifest['cases'] <= 4096:
        raise ValueError('Unexpected manifest modes or case count')
    args.output.mkdir(parents=True, exist_ok=False)

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def esptool(arguments, log, timeout=120):
        process = subprocess.run([sys.executable, '-m', 'esptool', '--chip', 'esp32p4',
                                  '--port', args.port, *arguments],
                                 capture_output=True, text=True, timeout=timeout)
        (args.output / log).write_text(process.stdout + process.stderr)
        process.check_returncode()
        return process.stdout

    def flash(path, log):
        esptool(['--baud', '1500000', '--before', 'default_reset', '--after', 'hard_reset',
                 'write_flash', hex(args.app_offset), str(path)], log)

    before = status()
    if before['app_sha256'] != args.expected_app_sha256 or before['ota']['running'] != args.slot:
        raise ValueError('Running app or slot does not match original development image')
    if before['voice']['active'] or before['ota']['busy'] or before['nyx']['state'] != 'up':
        raise ValueError('Development app is busy or disconnected')
    if args.expected_mac.lower() not in esptool(['chip_id'], 'chip-private.log', 30).lower():
        raise ValueError('USB chip MAC does not match development board')
    result = {'scope': 'Standalone classifier head numeric/resources only; no recognition acceptance',
              'probe_bin_sha256': args.expected_probe_sha256,
              'restore_bin_sha256': args.expected_restore_bin_sha256,
              'vectors_manifest_sha256': args.expected_vectors_manifest_sha256,
              'vectors': manifest, 'heads': [], 'accepted': False}

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Interrupted; restoring original development app')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        flash(args.probe_bin, 'probe-flash-private.log')
        with serial.Serial(args.port, 115200, timeout=.2) as uart, (args.output / 'uart-private.log').open('w') as log:
            uart.dtr = False
            uart.rts = True
            time.sleep(.1)
            uart.rts = False
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                line = uart.read_until(b'\n', 8192).decode('utf-8', errors='replace').strip()
                if line:
                    log.write(line + '\n')
                    log.flush()
                match = re.search(r'(PROBE_HEAD|P4_WAKE_PROBE_DONE) (\{.*\})', line)
                if match:
                    marker, data = match.groups()
                    value = json.loads(data)
                    print(match.group(0), flush=True)
                    if marker == 'PROBE_HEAD':
                        result['heads'].append(value)
                    else:
                        result['done'] = value
                        break
                if 'abort()' in line or 'Guru Meditation' in line:
                    raise RuntimeError('Probe firmware failed; see private UART log')
            validate_result(result, manifest)
            result['numeric_passed'] = True
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print('Restoring original development app', flush=True)
        try:
            flash(args.restore_app, 'restore-private.log')
            deadline = time.monotonic() + 60
            after = None
            while time.monotonic() < deadline:
                try:
                    after = status()
                    if (after['app_sha256'] == args.expected_app_sha256 and after['ota']['running'] == args.slot
                            and after['nyx']['state'] == 'up' and after['voice']['wake']['listening']
                            and after['voice']['service_id'] == before['voice']['service_id']):
                        result['restored'] = {'app_sha256': after['app_sha256'], 'original_service_verified': True,
                                              'nyx_state': 'up', 'hi_esp_listening': True}
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(1)
            if 'restored' not in result:
                raise RuntimeError('RESTORE VERIFICATION FAILED: original app, slot, service and Hi ESP required')
            print('Original app, slot, service and Hi ESP listening verified', flush=True)
        finally:
            (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
