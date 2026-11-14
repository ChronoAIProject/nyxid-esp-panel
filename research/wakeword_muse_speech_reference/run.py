"""Capture frozen consumed Muse PCM using the existing verified-slot transaction."""
import argparse
import json
from pathlib import Path
import signal
import sys
import time
import urllib.parse

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT.parent / 'wakeword_p4/experiments'))
import run_trained_p4_capture_mac_v1 as transaction
from fixtures import load, sha
from reduce import reduce_capture, validate_ready, validate_control_initialization, validate_control_result, CONTROL_FIELD

SCOPE = 'Standalone Muse DEFAULT complete consumed108 PCM reference only. Not live microphone, panel headroom, background FAR, voice connectivity or candidate acceptance.'


class Reader:
    def __init__(self, uart, log):
        self.uart, self.log = uart, log

    def next(self, timeout):
        deadline, pending = time.monotonic() + timeout, bytearray()
        while time.monotonic() < deadline:
            # A one-byte FIFO boundary is intentional during the baud handshake:
            # never read the new-baud READY before switching the host UART.
            data = self.uart.read(1)
            if not data:
                continue
            self.log.write(data)
            pending.extend(data)
            assert len(pending) <= 65536, 'UART line exceeds protocol bound'
            if data != b'\n':
                continue
            self.log.flush()
            line = bytes(pending).rstrip(b'\r\n')
            pending.clear()
            if any(marker in line for marker in (b'WAKE_BENCH_ERROR', b'Guru Meditation', b'abort()')):
                raise RuntimeError('Muse firmware error; preserve raw capture')
            if line.startswith(b'WAKE_BENCH_'):
                prefix, payload = line.decode('ascii').split(' ', 1)
                return prefix, json.loads(payload)
        raise TimeoutError('Muse UART message deadline; no restart')

    def expect(self, kind, timeout=30):
        actual, value = self.next(timeout)
        assert actual == kind, (kind, actual)
        return value


def write_all(uart, data):
    sent = 0
    while sent < len(data):
        written = uart.write(data[sent:sent + 4096])
        assert written is not None and written > 0
        sent += written
    # WCH tcdrain/flush can hang. Complete board samples/calls/CRC in RESULT,
    # checked before the next command, prove consumption of the entire payload.


class Backend(transaction.HardwareBackend):
    def __init__(self, cfg, references, payloads):
        super().__init__(cfg)
        self.references, self.payloads = references, payloads

    def capture(self, timeout):
        import serial
        path = self.cfg.output / 'uart-private.log'
        deadline = time.monotonic() + timeout
        with serial.Serial(self.cfg.port, 115200, timeout=.2, write_timeout=30) as uart, path.open('wb') as log:
            uart.dtr, uart.rts = False, True
            time.sleep(.1)
            uart.reset_input_buffer()
            uart.rts = False
            reader = Reader(uart, log)
            ready = reader.expect('WAKE_BENCH_READY')
            validate_ready(ready, 'wn10_himuse', True, True)
            assert ready['expanded_model_bytes'] == 430762
            baud = self.references['uart_baud']
            if baud != 115200:
                write_all(uart, f'BAUD {baud}\n'.encode())
                assert reader.expect('WAKE_BENCH_BAUD') == dict(baud=baud)
                uart.baudrate = baud
                assert reader.expect('WAKE_BENCH_READY') == ready
            for index, (entry, data) in enumerate(zip(self.references['fixtures']['rows'], self.payloads, strict=True)):
                row = dict(entry['original'], **{CONTROL_FIELD: None})
                for repeat in range(2):
                    assert time.monotonic() < deadline, 'Complete capture deadline; no restart'
                    clip_id = index * 2 + repeat
                    write_all(uart, f"CLIP {clip_id} {row['samples']} DEFAULT\n".encode())
                    prepared = reader.expect('WAKE_BENCH_CLIP_READY')
                    validate_control_initialization(prepared, row, clip_id)
                    write_all(uart, data)
                    events = []
                    while True:
                        kind, result = reader.next(min(30, max(.1, deadline - time.monotonic())))
                        if kind == 'WAKE_BENCH_EVENT':
                            events.append(result)
                        else:
                            assert kind == 'WAKE_BENCH_RESULT'
                            break
                    validate_control_result(result, events, row, clip_id, 512)
                    assert result['crc32'] == entry['pcm_crc32']
                    print(json.dumps(dict(phase='muse_pcm', completed=clip_id + 1, total=216)), flush=True)
            # This command is handled only after the final free/destroy. It
            # acknowledges completion without another detector invocation.
            write_all(uart, f'BAUD {baud}\n'.encode())
            assert reader.expect('WAKE_BENCH_BAUD') == dict(baud=baud)
            assert reader.expect('WAKE_BENCH_READY') == ready
        return path.read_bytes()


def freeze(build_path, fixtures_path):
    build, fixtures = json.loads(build_path.read_text()), json.loads(fixtures_path.read_text())
    assert build['schema'] == 'official-muse-consumed-speech-build-v1'
    assert fixtures['schema'] == 'muse-consumed108-fixtures-v1' and len(fixtures['rows']) == 108
    sources = dict(build['sources'])
    for path in list(PROJECT.glob('*.py')) + [PROJECT.parent / 'wakeword_baseline/run.py', Path(transaction.__file__)]:
        sources[str(path.relative_to(REPO))] = sha(path.read_bytes())
    assert all(sha((REPO / name).read_bytes()) == digest for name, digest in sources.items()), 'Source no longer matches built/prospective protocol'
    return dict(schema='muse-consumed108-frozen-protocol-v1', sources=sources,
        build_report_sha256=sha(build_path.read_bytes()), fixtures_report_sha256=sha(fixtures_path.read_bytes()),
        image_sha256=build['firmware_sha256'], elf_sha256=build['elf_sha256'],
        uart_baud=921600, fixtures=fixtures,
        prime_calls=63, chunk=512, sample_rate=16000, channels=1, sample_format='original PCM16 little endian',
        lifecycle='Unchanged baseline.c: per clip create,63zero prime,whole PCM with final-only padding,destroy; two adjacent fresh lifecycles per original row.',
        final_destroy_ack='Existing same-baud BAUD/READY command after final RESULT; no extra inference.',
        scope=SCOPE, candidate_qualified=False, threshold_selected=False, source_labels_changed=False,
        microphone_speaker_used=False, fresh_development_accessed=False, acceptance_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    for name in ('build-report', 'fixture-report', 'protocol'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('build-root', 'wave-root', 'output', 'restore-app'):
        parser.add_argument('--' + name, type=Path)
    for name in ('port', 'expected-mac', 'status-url', 'expected-app-sha256', 'expected-restore-sha256', 'slot'):
        parser.add_argument('--' + name)
    for name in ('app-offset', 'slot-bytes', 'partition-table-offset'):
        parser.add_argument('--' + name, type=lambda value: int(value, 0))
    args = parser.parse_args()
    plan = freeze(args.build_report, args.fixture_report)
    if args.action == 'freeze':
        with args.protocol.open('x') as stream:
            stream.write(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(dict(protocol_sha256=sha(args.protocol.read_bytes()), clips=108, executions=216, native_run=False)))
        return
    assert plan == json.loads(args.protocol.read_text()), 'Protocol/source changed after freeze'
    required = ('build_root', 'wave_root', 'output', 'restore_app', 'port', 'expected_mac', 'status_url',
                'expected_app_sha256', 'expected_restore_sha256', 'slot', 'app_offset', 'slot_bytes', 'partition_table_offset')
    assert all(getattr(args, name) is not None for name in required)
    url = urllib.parse.urlsplit(args.status_url)
    assert url.scheme == 'http' and url.hostname and not url.username and not url.password
    transaction.usb_serial_port(args.port)
    image = (args.build_root / 'build/nyx_muse_speech_reference.bin').read_bytes()
    assert sha(image) == plan['image_sha256']
    assert transaction.app_elf_sha256(image) == plan['elf_sha256']
    payloads = load(plan['fixtures'], args.wave_root)
    args.output.mkdir(exist_ok=False)
    (args.output / 'references.json').write_text(json.dumps(plan, indent=2) + '\n')
    cfg = transaction.Settings(args.port, args.expected_mac, args.status_url, args.expected_app_sha256,
        args.restore_app, args.expected_restore_sha256, args.app_offset, args.slot, args.slot_bytes,
        args.partition_table_offset, args.output, 'app-only', 900, 1500000)
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('Interrupted; restoring original development app')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        transaction.run_transaction(cfg, Backend(cfg, plan, payloads), image, plan, plan,
            capture_reducer=reduce_capture, scope=SCOPE)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
