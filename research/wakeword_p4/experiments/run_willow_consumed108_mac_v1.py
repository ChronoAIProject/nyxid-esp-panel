"""Run unchanged official Willow on the same consumed108 PCM used for Muse.

Reuses the established UART transport, complete-event validators and full-slot
backup/restore transaction. No microphone, speaker, new media or threshold sweep.
"""
import argparse
import json
from pathlib import Path
import signal
import struct
import sys
import time
import urllib.parse
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MUSE = REPO / 'research/wakeword_muse_speech_reference'
sys.path.insert(0, str(MUSE))
import run as transport
from fixtures import admit, pcm, sha
from reduce import messages, validate_ready, validate_control_initialization, validate_control_result, CONTROL_FIELD

transaction = transport.transaction
ENGINE = 'wn9_heywillow_tts'
DEFAULT_BITS = '3f24dd2f'
EXPANDED_BYTES = 291040
SCOPE = 'Official Willow DEFAULT on original consumed108 PCM; standalone P4/QEMU reference diagnosis, not background FAR, acoustic parity, candidate or voice connectivity acceptance.'


def reduce_capture(raw, references):
    records = messages(raw)
    index = 0
    def take(kind):
        nonlocal index
        assert index < len(records) and records[index][0] == kind, (index, kind)
        value = records[index][1]
        index += 1
        return value
    ready = take('WAKE_BENCH_READY')
    validate_ready(ready, ENGINE, True, True)
    assert ready['expanded_model_bytes'] == EXPANDED_BYTES
    baud = references['uart_baud']
    if baud != 115200:
        assert take('WAKE_BENCH_BAUD') == dict(baud=baud)
        assert take('WAKE_BENCH_READY') == ready
    cases, repeats = [], []
    for entry in references['fixtures']['rows']:
        current = []
        for repeat in range(2):
            clip_id = len(cases)
            row = dict(entry['original'], **{CONTROL_FIELD: None})
            prepared = take('WAKE_BENCH_CLIP_READY')
            validate_control_initialization(prepared, row, clip_id)
            control = prepared['control']
            assert all(control[key] == DEFAULT_BITS for key in ('default_getter_f32_hex', 'getter_after_set_f32_hex', 'getter_after_prime_f32_hex'))
            events = []
            while index < len(records) and records[index][0] == 'WAKE_BENCH_EVENT':
                events.append(take('WAKE_BENCH_EVENT'))
            result = take('WAKE_BENCH_RESULT')
            validate_control_result(result, events, row, clip_id, 512)
            assert result['crc32'] == entry['pcm_crc32']
            assert 0 < result['max_us'] <= result['total_us'] <= result['calls'] * result['max_us']
            current.append(dict(id=clip_id, repeat=repeat, original=entry['original'], initialization=prepared, events=events, result=result))
            cases.append(current[-1])
        def logical(case):
            return dict(prime=case['initialization']['control']['prime_returns'],
                threshold=case['initialization']['control']['getter_after_prime_f32_hex'],
                result={k: v for k, v in case['result'].items() if k not in ('id', 'total_us', 'max_us')},
                events=[{k: v for k, v in event.items() if k != 'id'} for event in case['events']])
        repeats.append(logical(current[0]) == logical(current[1]))
    assert take('WAKE_BENCH_BAUD') == dict(baud=baud)
    assert take('WAKE_BENCH_READY') == ready
    assert index == len(records)
    groups = []
    for literal in references['fixtures']['selected_literals']:
        for repeat in range(2):
            selected = [c for c in cases if c['original']['literal'] == literal and c['repeat'] == repeat]
            groups.append(dict(literal=literal, repeat=repeat, clips=len(selected),
                clips_with_events=sum(bool(c['events']) for c in selected), total_events=sum(len(c['events']) for c in selected)))
    return dict(schema='willow-consumed108-uart-reduction-v1', capture_sha256=sha(raw),
        reference_binding_sha256=sha(json.dumps(references, sort_keys=True).encode()),
        ready=ready, cases=cases, repeat_exact=repeats, all_repeats_exact=all(repeats), groups=groups,
        input_invocations=sum(c['result']['calls'] for c in cases), prime_invocations=len(cases) * 63,
        max_detect_us=max(c['result']['max_us'] for c in cases),
        setter_called=False, threshold_selected=False, candidate_qualified=False, limits=SCOPE)


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
            reader = transport.Reader(uart, log)
            ready = reader.expect('WAKE_BENCH_READY')
            validate_ready(ready, ENGINE, True, True)
            assert ready['expanded_model_bytes'] == EXPANDED_BYTES
            baud = self.references['uart_baud']
            transport.write_all(uart, f'BAUD {baud}\n'.encode())
            assert reader.expect('WAKE_BENCH_BAUD') == dict(baud=baud)
            uart.baudrate = baud
            assert reader.expect('WAKE_BENCH_READY') == ready
            total = len(self.payloads) * 2
            for index, (entry, data) in enumerate(zip(self.references['fixtures']['rows'], self.payloads, strict=True)):
                row = dict(entry['original'], **{CONTROL_FIELD: None})
                for repeat in range(2):
                    assert time.monotonic() < deadline, 'Capture deadline; preserve original run, no restart'
                    clip_id = index * 2 + repeat
                    transport.write_all(uart, f"CLIP {clip_id} {row['samples']} DEFAULT\n".encode())
                    prepared = reader.expect('WAKE_BENCH_CLIP_READY')
                    validate_control_initialization(prepared, row, clip_id)
                    transport.write_all(uart, data)
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
                    print(json.dumps(dict(phase='willow_consumed_pcm', completed=clip_id + 1, total=total)), flush=True)
            transport.write_all(uart, f'BAUD {baud}\n'.encode())
            assert reader.expect('WAKE_BENCH_BAUD') == dict(baud=baud)
            assert reader.expect('WAKE_BENCH_READY') == ready
        return path.read_bytes()


def freeze(args):
    build = json.loads(args.build_report.read_text())
    product = build['products'][ENGINE]
    image = args.probe_bin.read_bytes()
    assert len(image) == product['files']['nyx_wake_baseline.bin']['bytes']
    assert sha(image) == product['files']['nyx_wake_baseline.bin']['sha256']
    assert transaction.app_elf_sha256(image) == product['files']['nyx_wake_baseline.elf']['sha256']
    packed = (args.probe_bin.parent / 'esp-idf/main/baseline-hey-willow.bin').read_bytes()
    assert sha(packed) == product['files']['baseline-hey-willow.bin']['sha256'] and image.count(packed) == 1
    assert len(zlib.decompress(packed[4:])) == struct.unpack_from('<I', packed)[0] == EXPANDED_BYTES
    fixtures = admit(args.source_manifest, args.wave_root)
    fixtures.update(schema='willow-consumed108-fixtures-v1', evaluation_target_literal='Hey Willow')
    dependencies = [Path(__file__), Path(transport.__file__), MUSE / 'fixtures.py', MUSE / 'reduce.py',
        REPO / 'research/wakeword_baseline/run.py', Path(transaction.__file__), Path(transaction.reducer.__file__)]
    return dict(schema='willow-consumed108-frozen-protocol-v1',
        sources={str(p.relative_to(REPO)): sha(p.read_bytes()) for p in dependencies},
        build_report_sha256=sha(args.build_report.read_bytes()), historical_build_sources=build['sources'],
        image_sha256=sha(image), elf_sha256=transaction.app_elf_sha256(image),
        packed_model_sha256=sha(packed), expanded_model_sha256=sha(zlib.decompress(packed[4:])),
        expanded_model_bytes=EXPANDED_BYTES, fixtures=fixtures,
        uart_baud=921600, prime_calls=63, chunk=512, default_getter_f32_hex=DEFAULT_BITS,
        lifecycle='Per original row: create,63zero prime,complete original PCM,final-only512padding,destroy; two adjacent fresh lifecycles.',
        startup_sync='Reuse validated first-complete-READY synchronization; retain boot prefix bytes.',
        final_destroy_ack='Existing same-baud BAUD/READY after final RESULT; no additional inference.',
        scope=SCOPE, threshold_selected=False, candidate_qualified=False, microphone_speaker_used=False,
        fresh_development_accessed=False, acceptance_accessed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'reduce'))
    for name in ('build-report', 'probe-bin', 'source-manifest', 'wave-root', 'protocol', 'output', 'capture', 'restore-app'):
        parser.add_argument('--' + name, type=Path)
    for name in ('port', 'expected-mac', 'status-url', 'expected-app-sha256', 'expected-restore-sha256', 'slot'):
        parser.add_argument('--' + name)
    for name in ('app-offset', 'slot-bytes', 'partition-table-offset'):
        parser.add_argument('--' + name, type=lambda x: int(x, 0))
    args = parser.parse_args()
    if args.action == 'reduce':
        result = reduce_capture(args.capture.read_bytes(), json.loads(args.protocol.read_text()))
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
        print(json.dumps({k: result[k] for k in ('input_invocations', 'all_repeats_exact', 'groups')}))
        return
    assert all(getattr(args, name) is not None for name in ('build_report', 'probe_bin', 'source_manifest', 'wave_root', 'protocol'))
    plan = freeze(args)
    if args.action == 'freeze':
        with args.protocol.open('x') as stream:
            stream.write(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(dict(protocol_sha256=sha(args.protocol.read_bytes()), clips=108, executions=216, device_operated=False)))
        return
    assert plan == json.loads(args.protocol.read_text()), 'Changed protocol, build, source or PCM after freeze'
    required = ('output', 'restore_app', 'port', 'expected_mac', 'status_url', 'expected_app_sha256', 'expected_restore_sha256', 'slot', 'app_offset', 'slot_bytes', 'partition_table_offset')
    assert all(getattr(args, name) is not None for name in required)
    url = urllib.parse.urlsplit(args.status_url)
    assert url.scheme == 'http' and url.hostname and not url.username and not url.password
    transaction.usb_serial_port(args.port)
    payloads = [pcm(args.wave_root / e['local_path'], e['original']) for e in plan['fixtures']['rows']]
    assert all(zlib.crc32(p) == e['pcm_crc32'] for p, e in zip(payloads, plan['fixtures']['rows'], strict=True))
    args.output.mkdir(exist_ok=False)
    (args.output / 'references.json').write_text(json.dumps(plan, indent=2) + '\n')
    cfg = transaction.Settings(args.port, args.expected_mac, args.status_url, args.expected_app_sha256,
        args.restore_app, args.expected_restore_sha256, args.app_offset, args.slot, args.slot_bytes,
        args.partition_table_offset, args.output, 'app-only', 900, 1500000)
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('Interrupted; restoring original development app')
    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        transaction.run_transaction(cfg, Backend(cfg, plan, payloads), args.probe_bin.read_bytes(), plan, plan,
            capture_reducer=reduce_capture, scope=SCOPE)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    main()
