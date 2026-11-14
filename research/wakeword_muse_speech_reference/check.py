"""Constructed protocol/partial-UART checks; never opens physical hardware."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import wave
import zlib

from fixtures import sha, pcm, LITERALS
from reduce import reduce_capture, DEFAULT_BITS

spec = importlib.util.spec_from_file_location('muse_capture_runner', Path(__file__).with_name('run.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def encode(records):
    return b''.join((key + ' ' + json.dumps(value) + '\r\n').encode() for key, value in records)


def construct():
    value = struct.unpack('>f', bytes.fromhex(DEFAULT_BITS))[0]
    heap = dict(internal_free=500000, psram_free=30000000, internal_largest=400000, psram_largest=29000000)
    ready = dict(engine='wn10_himuse', mode='DET_MODE_95', rate=16000, resource_probe=True,
                 control_protocol='official-threshold-v1', expanded_model_bytes=430762,
                 heap_before_model=heap, heap_after_model=heap)
    entries, payloads = [], []
    for index in range(108):
        data = bytes([index % 256, 0]) * 1025
        payloads.append(data)
        row = dict(samples=1025, literal=LITERALS[index // 36], source_id=str(index), variant=0,
                   sha256='constructed', pcm_sha256=sha(data))
        entries.append(dict(original=row, pcm_crc32=zlib.crc32(data)))
    refs = dict(uart_baud=921600, fixtures=dict(rows=entries, selected_literals=list(LITERALS)))
    records = [('WAKE_BENCH_READY', ready), ('WAKE_BENCH_BAUD', dict(baud=921600)), ('WAKE_BENCH_READY', ready)]
    for clip_id in range(216):
        row = entries[clip_id // 2]
        control = dict(schema='official-threshold-v1', requested_f32_hex=None, setter_called=False,
            setter_result=None, default_getter=value, default_getter_f32_hex=DEFAULT_BITS,
            getter_after_set=value, getter_after_set_f32_hex=DEFAULT_BITS,
            getter_after_prime_f32_hex=DEFAULT_BITS, prime_returns=[0] * 63)
        prepared = dict(id=clip_id, chunk=512, threshold=value, control=control,
            create_us=1, prime_us=63, heap_before_create=heap, heap_after_create=heap, heap_after_prime=heap)
        records.append(('WAKE_BENCH_CLIP_READY', prepared))
        events = []
        if row['original']['literal'] == 'Hey Muse':
            events = [dict(id=clip_id, call=2, sample_end=1024, chunk_end=1024, result=1)]
            records.append(('WAKE_BENCH_EVENT', events[0]))
        result = dict(id=clip_id, complete=True, samples=1025, crc32=row['pcm_crc32'], events=len(events),
            calls=3, total_us=3, max_us=1, offsets=[1024] if events else [], events_emitted=len(events),
            offsets_complete=True, no_detect_calls=3-len(events), channel_verified_calls=0)
        records.append(('WAKE_BENCH_RESULT', result))
    records += [('WAKE_BENCH_BAUD', dict(baud=921600)), ('WAKE_BENCH_READY', ready)]
    return refs, payloads, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    refs, payloads, records = construct()
    raw = encode(records)
    result = reduce_capture(raw, refs)
    assert result['input_invocations'] == 648 and result['all_repeats_exact']
    assert result['prime_invocations'] == 13608 and len(result['cases']) == 216
    controls = []
    def rejected(name, altered):
        try:
            reduce_capture(altered, refs)
        except (AssertionError, ValueError, KeyError, RuntimeError):
            controls.append(name)
        else:
            raise AssertionError('Corrupt capture accepted: ' + name)
    rejected('truncated_line', raw[:-1])
    rejected('firmware_failure', raw + b'WAKE_BENCH_ERROR {}\n')
    rejected('missing_final_destroy_barrier', encode(records[:-2]))
    rejected('duplicate_initial_ready', encode(records[:1] + records))
    rejected('missing_result', encode([r for r in records if not (r[0] == 'WAKE_BENCH_RESULT' and r[1]['id'] == 0)]))
    for key, changed in [('crc32', 0), ('samples', 1024), ('complete', False), ('calls', 2), ('events_emitted', 0)]:
        altered = copy.deepcopy(records)
        next(value for kind, value in altered if kind == 'WAKE_BENCH_RESULT')[key] = changed
        rejected('result_' + key, encode(altered))
    for key, changed in [('default_getter_f32_hex', '3f000000'), ('setter_called', True), ('prime_returns', [0]*62)]:
        altered = copy.deepcopy(records)
        next(value for kind, value in altered if kind == 'WAKE_BENCH_CLIP_READY')['control'][key] = changed
        rejected('control_' + key, encode(altered))
    altered = copy.deepcopy(records)
    next(value for kind, value in altered if kind == 'WAKE_BENCH_EVENT')['sample_end'] = 1023
    rejected('event_sample_clock', encode(altered))
    rejected('extra_trailing_record', encode(records + records[-2:]))
    # A genuine repeat difference is retained and reported, not filtered away.
    altered = copy.deepcopy(records)
    next(value for kind, value in altered if kind == 'WAKE_BENCH_CLIP_READY' and value['id'] == 1)['control']['prime_returns'][0] = -1
    repeat_result = reduce_capture(encode(altered), refs)
    assert not repeat_result['all_repeats_exact'] and repeat_result['repeat_exact'].count(False) == 1

    # Drive the actual capture writer/reader with fractional writes and one-byte
    # incoming frames. This UART cannot flush, enumerate, or touch a real device.
    class Serial:
        closed = False
        def __init__(self, *args, **kwargs):
            self.incoming = bytearray(encode(records[:1]))
            self.command = bytearray()
            self.received = bytearray()
            self.clip_id = None
            self.completed = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            Serial.closed = True
        def reset_input_buffer(self):
            pass
        def read(self, size):
            assert size == 1
            data = bytes(self.incoming[:1]); del self.incoming[:1]
            return data
        def write(self, data):
            part = data[:257]
            if self.clip_id is not None:
                self.received.extend(part)
                expected = payloads[self.clip_id // 2]
                assert bytes(self.received) == expected[:len(self.received)]
                if len(self.received) == len(expected):
                    selected = [(key, value) for key, value in records if key in ('WAKE_BENCH_EVENT', 'WAKE_BENCH_RESULT') and value['id'] == self.clip_id]
                    self.incoming.extend(encode(selected)); self.clip_id = None; self.completed += 1
            else:
                self.command.extend(part)
                if self.command.endswith(b'\n'):
                    line = self.command.decode().strip(); self.command.clear()
                    if line.startswith('BAUD'):
                        if self.completed:
                            assert self.completed == 216
                            self.incoming.extend(encode(records[-2:]))
                        else:
                            self.incoming.extend(encode(records[1:3]))
                    else:
                        _, clip_id, samples, control = line.split()
                        assert control == 'DEFAULT' and int(samples) == 1025 and int(clip_id) == self.completed
                        self.clip_id = int(clip_id); self.received.clear()
                        self.incoming.extend(encode([row for row in records if row[0] == 'WAKE_BENCH_CLIP_READY' and row[1]['id'] == self.clip_id]))
            return len(part)
    saved = sys.modules.get('serial')
    sys.modules['serial'] = SimpleNamespace(Serial=Serial)
    with tempfile.TemporaryDirectory(prefix='muse-constructed-') as temporary:
        try:
            captured = runner.Backend(SimpleNamespace(port='CONSTRUCTED', output=Path(temporary)), refs, payloads).capture(30)
            assert captured == raw and Serial.closed
        finally:
            if saved is None:
                del sys.modules['serial']
            else:
                sys.modules['serial'] = saved
        path = Path(temporary) / 'zero.wav'
        with wave.open(str(path), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(payloads[0])
        row = dict(samples=1025, sha256=sha(path.read_bytes()), pcm_sha256=sha(payloads[0]))
        assert pcm(path, row) == payloads[0]
        try:
            pcm(path, dict(row, pcm_sha256='0'*64))
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong PCM identity accepted')
    report = dict(schema='muse-consumed108-constructed-checks-v1', corrupted_captures_rejected=controls,
        complete216_constructed_capture_exact=True, partial_writes_and_split_reads_checked=True,
        capture_port_closed=True, final_destroy_barrier_checked=True, repeat_difference_retained=True,
        wav_container_and_pcm_identity_checked=True, source_sha256=sha(Path(__file__).read_bytes()),
        physical_hardware_accessed=False, recognition_measured=False,
        limits='Invented input/events/timings only; UART/restore mechanics and corruption controls, not model or recognition evidence.')
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(corruptions_rejected=len(controls), capture_exact=True, hardware_accessed=False)))


if __name__ == '__main__':
    main()
