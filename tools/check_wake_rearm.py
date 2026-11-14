#!/usr/bin/env python3
"""Bounded idle wake/AEC rearm test on an explicitly identified development board.

Requires pyserial. Uses the private monitor config, but never writes monitor
state. Only UART mute/unmute commands mutate the board; neither writes NVS.
Raw UART stays in the private output directory and must not be published.
"""
import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

from monitor_stability import MAX_RESPONSE, config_read, sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--port', required=True)
    p.add_argument('--usb-serial', required=True)
    p.add_argument('--vid', type=int, required=True)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--cycles', type=int, default=10)
    p.add_argument('--allow-adapter-reset', action='store_true',
                   help='Allow one open-induced reset before the test baseline; never during cycles')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.cycles <= 100:
        p.error('cycles must be between 1 and 100')
    import serial
    import serial.tools.list_ports

    config = config_read(args.config)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    uart_log = args.output / 'private-uart.bin'
    uart_log.touch(mode=0o600, exist_ok=False)
    records = []
    baseline = None
    baseline_at = None
    settings = None
    identity_verified = False
    attempted_mute = False
    uart = None
    result = dict(completed=False, requested_cycles=args.cycles, completed_cycles=0,
                  scope='Idle AEC/audio-lease rearm; not calls or WakeNet recreation',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def status(label, accept_adapter_boot=False):
        nonlocal baseline, baseline_at, settings
        with urllib.request.urlopen(config['status_url'], timeout=4) as r:
            raw = r.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise RuntimeError('Oversized status response')
        s = json.loads(raw)
        now = time.monotonic()
        row = sample(s, config['expected_slug'], time.time())
        v = s['voice']
        stable_settings = {
            'app': s['app_sha256'], 'profile': s['update']['profile'],
            'agent': v['agent_id'], 'service': v['service_id'],
            'volume': v['volume'], 'gain': v['playback_gain'],
            'mode': v['input_mode'], 'preferred_mode': v['preferred_input_mode'],
            'greeting': v['greeting_enabled'], 'wake': v['wake']['enabled'],
            'background': s['lcd']['default_image'],
        }
        if baseline is None:
            baseline, baseline_at, settings = row, now, stable_settings
        elif stable_settings != settings:
            raise RuntimeError('Board configuration changed; aborting')
        elif abs((row['uptime_s'] - baseline['uptime_s']) - (now - baseline_at)) > 5:
            if not accept_adapter_boot:
                raise RuntimeError('Board rebooted; aborting')
            result['adapter_open_reset'] = dict(previous_uptime_s=baseline['uptime_s'],
                                              next_uptime_s=row['uptime_s'])
            baseline, baseline_at = row, now
        if (v['active'] or v['busy'] or s['ota']['busy'] or
                s['media']['mode'] != 'idle' or not row['network_connected']):
            raise RuntimeError('Board is no longer idle; aborting without stopping user work')
        if row['counters'] != baseline['counters']:
            raise RuntimeError('Voice/wake/media activity changed; aborting')
        row['label'] = label
        records.append(row)
        return row

    def read_uart(seconds):
        end = time.monotonic() + seconds
        data = bytearray()
        with uart_log.open('ab', buffering=0) as f:
            while time.monotonic() < end:
                block = uart.read(4096)
                if block:
                    f.write(block)
                    data.extend(block)
        return bytes(data)

    def command(text, expect):
        read_uart(.15)
        uart.write(text.encode('ascii') + b'\r')
        uart.flush()
        data = bytearray()
        end = time.monotonic() + 4
        while time.monotonic() < end:
            data.extend(read_uart(.2))
            if re.search(expect, data):
                return bytes(data)
        raise RuntimeError('UART command response deadline exceeded')

    def wait_phase(muted, label):
        end = time.monotonic() + 12
        while time.monotonic() < end:
            row = status(label)
            phase = row['phase']
            if (phase['microphone_muted'] == muted and
                    phase['wake_listening'] == (not muted) and
                    phase['wake_state'] == ('muted' if muted else 'listening')):
                return row
            time.sleep(.25)
        raise RuntimeError('Wake phase deadline exceeded')

    def stable_rows(label):
        rows = []
        for _ in range(3):
            row = status(label)
            if row['phase'] != baseline['phase']:
                raise RuntimeError('Listener changed during stable sampling')
            rows.append(row)
            time.sleep(.5)
        return {k: statistics.median(r['heap'][k] for r in rows) for k in rows[0]['heap']}

    try:
        ports = [x for x in serial.tools.list_ports.comports() if x.device == args.port]
        if len(ports) != 1 or (ports[0].vid, ports[0].pid, ports[0].serial_number) != (
                args.vid, args.pid, args.usb_serial):
            raise RuntimeError('USB identity mismatch')
        before = status('before_uart')
        if before['phase']['microphone_muted'] or not before['phase']['wake_listening']:
            raise RuntimeError('Requires an already unmuted idle listener')
        uart = serial.Serial(port=None, baudrate=115200, timeout=.1, exclusive=True)
        uart.port = args.port
        uart.dtr = uart.rts = False
        uart.open()
        if args.allow_adapter_reset:
            # Some USB adapters reset on open regardless of requested DTR/RTS.
            # Capture that boot once and let the existing app finish startup.
            read_uart(30)
        data = command('nyx-status', rb'slug=([^\s]+)')
        match = re.search(rb'slug=([^\s]+)', data)
        if not match or match.group(1).decode() != config['expected_slug']:
            raise RuntimeError('UART and HTTP identity mismatch')
        status('after_uart', accept_adapter_boot=args.allow_adapter_reset)
        identity_verified = True
        result['initial_heap'] = stable_rows('initial')
        cycle_heaps = result['cycle_heaps'] = []
        for cycle in range(1, args.cycles + 1):
            status(f'{cycle}:before_mute')
            attempted_mute = True
            command('nyx-voice mute', rb'local_muted=1\b')
            wait_phase(True, f'{cycle}:muted')
            command('nyx-voice unmute', rb'local_muted=0\b')
            wait_phase(False, f'{cycle}:listening')
            cycle_heaps.append(stable_rows(f'{cycle}:settled'))
            result['completed_cycles'] = cycle
            print(json.dumps({'cycle': cycle, 'heap': cycle_heaps[-1]}), flush=True)
        result['cycle_heaps'] = cycle_heaps
        result['final_heap'] = stable_rows('final')
        result['final_minus_initial_bytes'] = {
            k: result['final_heap'][k] - result['initial_heap'][k] for k in result['initial_heap']}
        result['completed'] = True
    except (Exception, KeyboardInterrupt) as e:
        result['error_type'] = type(e).__name__
        if type(e) is RuntimeError:
            result['error'] = str(e)  # Only the controlled messages above.
        # Do not serialize exception text: network/serial errors can contain
        # private addresses. Full UART is already retained privately.
    finally:
        try:
            if identity_verified and attempted_mute:
                row = status('restore_check')
                if row['phase']['microphone_muted']:
                    command('nyx-voice unmute', rb'local_muted=0\b')
                row = wait_phase(False, 'restored')
                result['restored_unmuted_listener'] = True
        except (Exception, KeyboardInterrupt) as e:
            result['restore_error_type'] = type(e).__name__
        if uart is not None:
            uart.close()
        for filename, value in [('samples.json', records), ('result.json', result)]:
            dest = args.output / filename
            dest.write_text(json.dumps(value, indent=2) + '\n')
            dest.chmod(0o600)
    print(json.dumps(result), flush=True)
    return 0 if result['completed'] and result.get('restored_unmuted_listener') else 1


if __name__ == '__main__':
    raise SystemExit(main())
