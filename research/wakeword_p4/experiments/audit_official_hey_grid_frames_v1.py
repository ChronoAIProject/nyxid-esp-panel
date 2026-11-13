"""Independently compare saved native detector frames, complete PCM and report.

Extract only detector JSON frames from a private UART log. No boot/flash log,
device identity, microphone, device access or model inference is exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import wave
import zlib


KINDS = ('READY', 'CLIP_READY', 'EVENT', 'RESULT')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(args):
    frames = []
    for line in args.uart_private.read_text().splitlines():
        for kind in KINDS:
            marker = 'WAKE_BENCH_' + kind + ' '
            at = line.find(marker)
            if at >= 0:
                frames.append(dict(kind=kind, value=json.loads(line[at + len(marker):])))
                break
    with args.output.open('x') as output:
        output.write(json.dumps(frames, indent=2) + '\n')
    print(json.dumps(dict(detector_frames=len(frames), frames_sha256=sha(args.output))))


def audit(args):
    frames = json.loads(args.frames.read_text())
    report = json.loads(args.report.read_text())
    manifest = args.fixtures / 'manifest.jsonl'
    rows = list(map(json.loads, manifest.read_text().splitlines()))
    assert sha(manifest) == report['manifest_sha256'] and len(rows) == len(report['clips']) == 156
    ready = [f['value'] for f in frames if f['kind'] == 'READY']
    assert len(ready) == 2 and ready[0] == ready[1] == report['firmware_ready']
    assert ready[0]['engine'] == 'wn9_heywillow_tts'
    assert ready[0]['mode'] == 'DET_MODE_95' and ready[0]['rate'] == 16000
    assert ready[0]['control_protocol'] == 'official-threshold-v1'
    stream = [f for f in frames if f['kind'] != 'READY']
    cursor = calls = positives = prime_returns = 0
    for index, (row, actual) in enumerate(zip(rows, report['clips'], strict=True)):
        assert stream[cursor] == dict(kind='CLIP_READY', value=actual['initialization'])
        cursor += 1
        events = []
        while cursor < len(stream) and stream[cursor]['kind'] == 'EVENT':
            events.append(stream[cursor]['value']); cursor += 1
        result = actual['result']
        assert stream[cursor] == dict(kind='RESULT', value=result)
        cursor += 1
        assert events == actual['raw_events']
        init = actual['initialization']; control = init['control']
        assert init['id'] == result['id'] == index and init['chunk'] == 512
        request = row['reference_threshold_f32_hex']
        assert control['requested_f32_hex'] == request
        assert control['setter_called'] is (request is not None)
        assert control['setter_result'] == (None if request is None else 1)
        for number, bits in [(control['default_getter'], control['default_getter_f32_hex']),
                             (control['getter_after_set'], control['getter_after_set_f32_hex']),
                             (init['threshold'], control['getter_after_prime_f32_hex'])]:
            assert 0 < number < 1 and struct.pack('>f', number).hex() == bits
        assert len(control['prime_returns']) == 63
        assert all(type(v) is int and v in (-1, 0, 1) for v in control['prime_returns'])
        path = args.fixtures / row['path']
        assert sha(path) == row['sha256']
        with wave.open(str(path), 'rb') as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) == (16000, 1, 2, 'NONE')
            assert wav.getnframes() == row['samples']
            data = wav.readframes(wav.getnframes())
        assert len(data) == row['samples'] * 2
        assert hashlib.sha256(data).hexdigest() == row['pcm_sha256']
        expected_calls = (row['samples'] + 511) // 512
        assert result['complete'] is True and result['calls'] == expected_calls
        assert result['samples'] == row['samples'] and result['crc32'] == zlib.crc32(data)
        assert result['events'] == result['events_emitted'] == len(events)
        assert result['no_detect_calls'] + result['channel_verified_calls'] + len(events) == expected_calls
        assert min(result['no_detect_calls'], result['channel_verified_calls']) >= 0
        assert result['offsets_complete'] is (len(events) <= 64)
        assert result['offsets'] == [e['sample_end'] for e in events[:64]]
        previous_call = 0
        for event in events:
            assert event['id'] == index and event['result'] == 1
            assert previous_call < event['call'] <= expected_calls
            assert event['chunk_end'] == event['call'] * 512
            assert event['sample_end'] == min(event['chunk_end'], row['samples'])
            previous_call = event['call']
        calls += expected_calls; positives += len(events); prime_returns += 63
    assert cursor == len(stream)
    result = dict(schema='official-hey-consumed-speech-grid-saved-frame-audit-v1',
        frames_sha256=sha(args.frames), original_report_sha256=sha(args.report),
        streams=156, input_calls=calls, positive_event_frames=positives, prime_returns=prime_returns,
        all_frame_order_and_complete_PCM_CRC_samples_calls_getters_verified=True,
        scope='Independent saved-UART/complete-fixture comparison. Not second inference, fresh recognition, FAR or acceptance.')
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('extract')
    prep.add_argument('--uart-private', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    check = commands.add_parser('audit')
    for name in ('frames', 'report', 'fixtures', 'output'):
        check.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    extract(args) if args.command == 'extract' else audit(args)
