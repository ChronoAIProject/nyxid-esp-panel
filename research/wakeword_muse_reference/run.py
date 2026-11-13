"""Guarded Muse zero-input probe; only the verified active app slot is written."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import urllib.request
import zlib

import serial


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_cases(ready, cases, done):
    assert ready['schema'] == done['schema'] == 'muse-zero-v1'
    assert ready['engine'] == 'wn10_himuse' and ready['mode'] == 'DET_MODE_95'
    assert ready['expanded_model_bytes'] == 817708
    assert len(cases) == done['cases'] == 5
    chunk = cases[0]['chunk']
    assert type(chunk) is int and 1 < chunk <= 4096
    counts = [1, chunk - 1, chunk + 1, 16000, 16001]
    prime_events = input_events = 0
    first_words = cases[0]['words']
    for index, (case, samples) in enumerate(zip(cases, counts, strict=True)):
        assert case['id'] == index and case['samples'] == samples
        assert case['chunk'] == chunk and case['rate'] == 16000 and case['channels'] == 1
        calls = (samples + chunk - 1) // chunk
        assert case['calls'] == calls and case['crc32'] == zlib.crc32(bytes(samples * 2))
        assert case['words'] == first_words and 1 <= len(first_words) <= 8
        for i, word in enumerate(case['words'], 1):
            assert word['index'] == i and isinstance(word['name'], str) and word['name']
            assert word['default_bits'] == word['after_prime_bits']
            value = struct.unpack('>f', bytes.fromhex(word['default_bits']))[0]
            assert math.isfinite(value) and 0 < value < 1
        assert len(case['prime_returns']) == 63 and len(case['input_returns']) == calls
        for values in (case['prime_returns'], case['input_returns']):
            assert all(type(v) is int and -1 <= v <= len(first_words) for v in values)
        prime_events += sum(v > 0 for v in case['prime_returns'])
        input_events += sum(v > 0 for v in case['input_returns'])
        assert 0 <= case['detect_max_us'] <= case['detect_total_us']
        assert case['create_us'] >= 0 and case['prime_us'] >= 0
        assert case['probe_buffer_bytes'] == chunk * 2 + calls * 4
    return dict(schema='muse-zero-native-summary-v1', chunk=chunk, rate=16000, channels=1,
        word_metadata=first_words, cases=5, input_calls=sum(c['calls'] for c in cases),
        prime_calls=315, prime_positive_events=prime_events, input_positive_events=input_events,
        zero_positive_events_absent=prime_events == input_events == 0,
        complete_zero_sample_call_crc_return_accounting=True, post_prime_getters_stable=True,
        recognition_accepted=False, equal_far_comparison=False, threshold_selected=False,
        scope='Five internally generated zero-PCM streams; initialization/resources only, not speech recognition, FAR or voice acceptance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--expected-mac', required=True)
    parser.add_argument('--status-url', required=True)
    parser.add_argument('--expected-app-sha256', required=True)
    parser.add_argument('--restore-app', type=Path, required=True)
    parser.add_argument('--restore-sha256', required=True)
    parser.add_argument('--probe-bin', type=Path, required=True)
    parser.add_argument('--probe-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--build-report', type=Path, required=True)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--app-offset', type=lambda s: int(s, 0), required=True)
    parser.add_argument('--slot-bytes', type=lambda s: int(s, 0), required=True)
    parser.add_argument('--restore-readiness', choices=('app-only', 'online'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert args.status_url.startswith('http://')
    plan = json.loads(args.protocol.read_text())
    build = json.loads(args.build_report.read_text())
    assert plan['schema'] == 'official-muse-zero-plan-v1'
    assert plan['firmware_sha256'] == build['firmware_sha256'] == args.probe_sha256
    assert plan['build_report_sha256'] == sha(args.build_report)
    assert plan['sources'] == build['sources']
    repo = Path(__file__).resolve().parents[2]
    for name, digest in plan['sources'].items():
        assert sha(repo / name) == digest, name
    assert sha(args.restore_app) == args.restore_sha256 and sha(args.probe_bin) == args.probe_sha256
    assert bytes.fromhex(args.expected_app_sha256) in args.restore_app.read_bytes()
    assert all(0 < p.stat().st_size <= args.slot_bytes for p in (args.restore_app, args.probe_bin))
    owner = subprocess.run(['lsof', args.port], capture_output=True, text=True)
    assert owner.returncode == 1 and not owner.stdout and not owner.stderr

    def status():
        with urllib.request.urlopen(args.status_url, timeout=5) as response:
            return json.load(response)

    def ready(value, before):
        return (value['app_sha256'] == args.expected_app_sha256 and value['ota']['running'] == args.slot
            and not value['ota']['busy'] and not value['voice']['active'] and not value['voice'].get('busy', False)
            and value['voice']['wake']['listening'] and value['voice']['wake']['word'] == before['voice']['wake']['word']
            and all(value['voice'][key] == before['voice'][key] for key in ('service_id', 'agent_id'))
            and (args.restore_readiness == 'app-only' or value['nyx']['state'] == 'up'))

    def esp(arguments, name, timeout=180):
        result = subprocess.run([sys.executable, '-m', 'esptool', '--chip', 'esp32p4', '--port', args.port, *arguments],
            capture_output=True, text=True, timeout=timeout)
        (args.output / name).write_text(result.stdout + result.stderr)
        result.check_returncode()
        return result.stdout

    def flash(path, name):
        esp(['--baud', '1500000', '--before', 'default_reset', '--after', 'hard_reset',
             'write_flash', hex(args.app_offset), str(path)], name)

    before = status()
    assert ready(before, before)
    assert args.expected_mac.lower() in esp(['chip_id'], 'chip-private.log', 30).lower()
    partition = args.output / 'partition-private.bin'
    esp(['read_flash', '0x8000', '0x1000', str(partition)], 'partition-private.log', 30)
    entries = []
    raw = partition.read_bytes()
    for at in range(0, len(raw), 32):
        magic, kind, subtype, offset, size, label, flags = struct.unpack_from('<HBBII16sI', raw, at)
        if magic == 0x50aa:
            entries.append(dict(kind=kind, subtype=subtype, offset=offset, size=size,
                                label=label.rstrip(b'\0').decode('ascii'), flags=flags))
    selected = [e for e in entries if e['kind'] == 0 and e['label'] == args.slot]
    assert len(selected) == 1 and selected[0]['offset'] == args.app_offset
    assert selected[0]['size'] == args.slot_bytes and selected[0]['flags'] == 0
    # Re-read after esptool's resets before any probe write.
    deadline = time.monotonic() + 30
    while True:
        try:
            current = status()
            if ready(current, before):
                break
        except (OSError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Original app did not return after read-only chip/partition checks')
        time.sleep(1)
    report = dict(schema='muse-zero-guarded-native-v1', probe_sha256=args.probe_sha256,
        protocol_sha256=sha(args.protocol), build_report_sha256=sha(args.build_report),
        restore_sha256=args.restore_sha256, restore_readiness=args.restore_readiness,
        before=dict(app_sha256=before['app_sha256'], slot=before['ota']['running'], nyx_state=before['nyx']['state']),
        cases=[], audio_io_performed=False, candidate_promoted=False, acceptance_accessed=False)

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Interrupted; restoring original application')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        flash(args.probe_bin, 'probe-flash-private.log')
        with serial.Serial(args.port, 115200, timeout=.2) as uart, (args.output / 'uart-private.log').open('w') as log:
            uart.dtr = False
            uart.rts = True
            time.sleep(.1)
            uart.rts = False
            uart.reset_input_buffer()
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                line = uart.read_until(b'\n', 32768).decode('utf-8', errors='replace').strip()
                if line:
                    log.write(line + '\n')
                    log.flush()
                if 'abort()' in line or 'Guru Meditation' in line:
                    raise RuntimeError('Muse probe aborted; retain private UART evidence')
                for marker in ('MUSE_REFERENCE_READY', 'MUSE_REFERENCE_CASE', 'MUSE_REFERENCE_DONE'):
                    at = line.find(marker + ' ')
                    if at >= 0:
                        value = json.loads(line[at + len(marker) + 1:])
                        if marker.endswith('READY'):
                            assert 'firmware_ready' not in report
                            report['firmware_ready'] = value
                        elif marker.endswith('CASE'):
                            report['cases'].append(value)
                            print(json.dumps(dict(case=value['id'], chunk=value['chunk'], samples=value['samples'])), flush=True)
                        else:
                            report['done'] = value
                if 'done' in report:
                    break
            if 'done' not in report:
                raise TimeoutError('No terminal Muse frame')
        report['summary'] = audit_cases(report['firmware_ready'], report['cases'], report['done'])
    finally:
        print('Restoring original development app', flush=True)
        flash(args.restore_app, 'restore-private.log')
        deadline = time.monotonic() + 60
        while True:
            try:
                after = status()
                if ready(after, before):
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError('Original app restoration verification failed')
            time.sleep(1)
        report['restored'] = dict(app_sha256=after['app_sha256'], slot=after['ota']['running'],
            nyx_state=after['nyx']['state'], service_and_agent_unchanged=True, voice_idle=True,
            wake_word=after['voice']['wake']['word'], wake_listening=True,
            voice_connectivity_accepted=False)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('Original app, selected service/agent and wake listener restored; Nyx=' + after['nyx']['state'], flush=True)


if __name__ == '__main__':
    main()
